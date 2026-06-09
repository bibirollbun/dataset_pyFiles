import warnings
warnings.simplefilter('ignore')


import pandas as pd
import numpy as np

train_path = "/kaggle/input/playground-series-s5e12/train.csv"
test_path = "/kaggle/input/playground-series-s5e12/test.csv"
sample_submission_path = "/kaggle/input/playground-series-s5e12/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("âœ… Data loaded successfully.")


# Display the dimensions and head of the training data
print(f"\nTraining set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

print("\n--- Training Data Head (First 5 Rows) ---")
train_df.head()


print("\n--- Training Data Information ---")
train_df.info()


import pandas as pd

# df = train_df
df = test_df

# Identify categorical columns (object dtype)
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Print unique values for each categorical column
for col in cat_cols:
    print(f"\n--- {col} ---")
    print(df[col].unique())


train_df.head(10)


df_training_mapped = train_df.copy()

CAT_MAPS = {
    'gender': {
        'Female': 0,
        'Male': 1,
        'Other': 2
    },
    'ethnicity': {
        'Hispanic': 0,
        'White': 1,
        'Asian': 2,
        'Black': 3,
        'Other': 4
    },
    'education_level': {
        'No formal': 0,
        'Highschool': 1,
        'Graduate': 2,
        'Postgraduate': 3
    },
    'income_level': {
        'Low': 0,
        'Lower-Middle': 1,
        'Middle': 2,
        'Upper-Middle': 3,
        'High': 4
    },
    'smoking_status': {
        'Never': 0,
        'Former': 1,
        'Current': 2
    },
    'employment_status': {
        'Unemployed': 0,
        'Student': 1,
        'Employed': 2,
        'Retired': 3
    }
}

# Apply mappings
cat_cols = list(CAT_MAPS.keys())

for col in cat_cols:
    df_training_mapped[col] = df_training_mapped[col].map(CAT_MAPS[col])



df_training_mapped.head(10)


# --- Helper Function: Add Engineered Features ---
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-informed features to enhance diabetes risk prediction.
    Works on both train and test DataFrames.
    """
    df = df.copy()
    
    # Cardiovascular
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['bp_risk_score'] = (df['systolic_bp'] / 120.0) * (df['diastolic_bp'] / 80.0)
    
    # Lipid profiles
    df['cholesterol_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-6)
    df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-6)
    
    # Lifestyle balance
    df['activity_screen_ratio'] = df['physical_activity_minutes_per_week'] / (df['screen_time_hours_per_day'] + 1)
    
    # BMI indicators
    df['bmi_overweight'] = (df['bmi'] >= 25).astype(int)
    df['bmi_obese'] = (df['bmi'] >= 30).astype(int)
    
    return df

# --- Apply to Training Data ---
# Start from mapped training data (after CAT_MAPS applied)
train_processed = df_training_mapped.copy()
train_processed = add_engineered_features(train_processed)

# Prepare X and y
X = train_processed.drop(columns=['id', 'diagnosed_diabetes'])
y = train_processed['diagnosed_diabetes'].astype(int)

# --- Apply to Test Data ---
# Start from raw test_df (before any mapping)
df_test = test_df.copy()

# Apply categorical mappings
for col in CAT_MAPS.keys():
    df_test[col] = df_test[col].map(CAT_MAPS[col])
df_test[list(CAT_MAPS.keys())] = df_test[list(CAT_MAPS.keys())].fillna(-1).astype(int)

# Add engineered features
test_processed = add_engineered_features(df_test)

# Align test features with training
X_test = test_processed[X.columns]  # Ensures identical feature set and order


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

# --- Keep your existing preprocessing for X, y, and X_test ---
# (You already did: CAT_MAPS, feature engineering, X = ..., y = ..., X_test = ...)

# --- NEW: Cross-Validation Setup ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))          # Out-of-fold predictions (for CV score)
test_preds = np.zeros(len(X_test))    # Final test predictions (will be averaged)

print("Starting 5-Fold Cross-Validation...\n")

# --- CV Loop ---
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/5")
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Define a slightly simpler model to reduce overfitting
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,                 # Reduced from 6 â†’ less overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,                 # L1 regularization
        reg_lambda=1,                # L2 regularization
        eval_metric='auc',
        early_stopping_rounds=30,    # Faster stopping
        random_state=42
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=0  # Set to 50 if you want training logs
    )
    
    # Predict on validation (for OOF)
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    
    # Predict on test and accumulate (average later)
    test_preds += model.predict_proba(X_test)[:, 1] / 5
    
    # Print fold score
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"  â†’ Fold AUC: {fold_auc:.4f}\n")

# --- Final CV Score ---
cv_auc = roc_auc_score(y, oof_preds)
print(f"âœ… Final Cross-Validated AUC (OOF): {cv_auc:.4f}")




# --- Create Submission ---
submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})

submission.to_csv('submission_cv.csv', index=False)
print("âœ… Submission saved to 'submission_cv.csv'")

submission

