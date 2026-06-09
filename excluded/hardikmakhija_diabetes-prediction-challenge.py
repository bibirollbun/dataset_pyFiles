## CELL 1: Imports and Setup

# Install LightGBM if it's not present in the environment
!pip install lightgbm -q

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

print("âœ“ All libraries imported successfully.")

# Define the correct Kaggle file paths
BASE_PATH = "/kaggle/input/playground-series-s5e12" 
TRAIN_FILE = f"{BASE_PATH}/train.csv"
TEST_FILE = f"{BASE_PATH}/test.csv"
SUBMISSION_FILE = f"{BASE_PATH}/sample_submission.csv"

# --------------------------------------------------------------------------------------
## CELL 2: Data Loading, Cleaning, and Feature Engineering

print("--- Loading Data and Initial Cleaning ---")
start_time = time.time()

# 1. Load Data
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    sample_submission = pd.read_csv(SUBMISSION_FILE)
    print("âœ“ Data loaded successfully from Kaggle input path.")
except FileNotFoundError as e:
    print(f"\nâ�Œ FATAL ERROR: Data files not found at {BASE_PATH}. Please ensure your Kaggle dataset is correctly added.")
    raise

# Separate target variable and IDs
y = train_df['diagnosed_diabetes']
train_ids = train_df['id']
test_ids = test_df['id']

# Drop ID and target from features
X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
X_test = test_df.drop('id', axis=1)

# Check class balance
target_mean = y.mean()
print(f"Target Class Imbalance: {target_mean*100:.2f}% (Positive Class)")

# 2. Feature Engineering

def feature_engineer(df):
    # 2.1. Blood Pressure (Average and Pulse Pressure)
    df['mean_bp'] = (df['systolic_bp'] + df['diastolic_bp']) / 2
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    
    # 2.2. Cholesterol Ratios
    df['chol_ratio_hdl_ldl'] = df['hdl_cholesterol'] / (df['ldl_cholesterol'] + 1e-6)
    df['chol_ratio_total_hdl'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)

    # 2.3. Lifestyle Indicators
    df['activity_to_screen_ratio'] = df['physical_activity_minutes_per_week'] / (df['screen_time_hours_per_day'] * 60 + 1e-6) 
    
    # 2.4. Age and BMI interaction
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    
    # 2.5. Categorical Cleaning
    df['income_level'] = df['income_level'].str.replace('-', '_', regex=False)
    
    return df

X = feature_engineer(X)
X_test = feature_engineer(X_test)

print(f"Feature Engineering complete. New features: {X.shape[1] - test_df.shape[1] + 1}")
print(f"Time taken: {time.time() - start_time:.2f} seconds.")

# --------------------------------------------------------------------------------------
## CELL 3: Preprocessing Pipeline (Scaling and Encoding)

print("\n--- Building Preprocessing Pipeline ---")

# Define feature groups
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Binary history features should not be scaled
binary_history_features = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
for feature in binary_history_features:
    if feature in numerical_features:
        numerical_features.remove(feature)


# Create the preprocessing pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        # 1. Numerical Pipeline: Impute NaNs with median, then Standard Scale
        ('num', 
         Pipeline([
             ('imputer', SimpleImputer(strategy='median')),
             ('scaler', StandardScaler())
         ]), 
         numerical_features),
        
        # 2. Categorical Pipeline: Impute NaNs with mode, then One-Hot Encode
        ('cat', 
         Pipeline([
             ('imputer', SimpleImputer(strategy='most_frequent')),
             ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
         ]), 
         categorical_features),
         
        # 3. Binary features: Pass through without modification
        ('pass', 'passthrough', binary_history_features)
    ],
    remainder='drop'
)

# Fit and transform the data
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

print(f"Processed training features shape: {X_processed.shape}")
print(f"Processed test features shape: {X_test_processed.shape}")
print("âœ“ Preprocessing pipeline built and applied.")

# --------------------------------------------------------------------------------------
## CELL 4: Model Training (LightGBM with Stratified K-Fold)

print("\n--- Training LightGBM Model with 5-Fold Cross-Validation (Faster Config) ---")
start_time = time.time()

# Configuration
N_SPLITS = 5
RANDOM_SEED = 42
KFOLD = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# OPTIMIZED LightGBM Parameters for Speed
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,        # REDUCED from 3000
    'learning_rate': 0.05,       # INCREASED from 0.01 (Major Speedup)
    'num_leaves': 20, 
    'max_depth': 6,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'reg_alpha': 0.1,  
    'reg_lambda': 0.1, 
    'n_jobs': -1,              # Use all available cores
    'seed': RANDOM_SEED,
    'verbose': -1,
    'scale_pos_weight': (1 - target_mean) / target_mean 
}

# Initialize prediction arrays
oof_preds = np.zeros(X_processed.shape[0])
test_preds = np.zeros(X_test_processed.shape[0])
cv_scores = []

# Loop through each fold
for fold, (train_index, val_index) in enumerate(KFOLD.split(X_processed, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    
    X_train, X_val = X_processed[train_index], X_processed[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Initialize and train LightGBM model
    model = lgb.LGBMClassifier(**lgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=-1)]
    )
    
    # Make predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_index] = val_preds
    
    # Calculate and store fold AUC score
    fold_auc = roc_auc_score(y_val, val_preds)
    cv_scores.append(fold_auc)
    print(f"Fold {fold+1} AUC: {fold_auc:.6f}")
    
    # Accumulate test predictions
    test_preds += model.predict_proba(X_test_processed)[:, 1] / N_SPLITS

# Final CV Score
mean_cv_auc = np.mean(cv_scores)
print(f"\nâœ… FINAL CV AUC Score: {mean_cv_auc:.6f}")
print(f"Time taken for training: {time.time() - start_time:.2f} seconds.")

# --------------------------------------------------------------------------------------
## CELL 5: Submission File Generation

print("\n--- Generating Submission File ---")

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'diagnosed_diabetes': test_preds
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("âœ“ submission.csv created successfully.")
print("--- Sample Submission Head ---")
print(submission_df.head())
print("\n" + "=" * 60)
print("ğŸ�‰ DIABETES PREDICTION SOLUTION READY! (Optimized for speed)")
print("Next: Download and upload submission.csv to the Kaggle competition.")
print("=" * 60)

