import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Modeling
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Algorithms
import xgboost as xgb
import lightgbm as lgb

# Settings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

print("Libraries imported successfully.")


# Load datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv') # If available, otherwise we build it later

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Combine for consistent preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['loan_paid_back'] = np.nan # Placeholder

df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Preview
df.head()


def preprocess_data(dataframe):
    df_proc = dataframe.copy()
    
    # 1. Encode 'grade_subgrade' (Ordinal)
    # We assume A1 is better/higher than D5. We can map them to numbers.
    # Get all unique grades, sort them alphabetically (A1, A2... D5), then map to integers.
    grades = sorted(df_proc['grade_subgrade'].dropna().unique())
    grade_map = {grade: i for i, grade in enumerate(grades)}
    df_proc['grade_encoded'] = df_proc['grade_subgrade'].map(grade_map)
    
    # Extract just the letter grade (A, B, C, D) as a separate feature
    df_proc['grade_letter'] = df_proc['grade_subgrade'].str[0]
    
    # 2. Encode 'education_level' (Ordinal)
    # We manually define the order of education
    edu_map = {
        "High School": 0, 
        "Bachelor's": 1, 
        "Master's": 2,
        "PhD": 3 # Including just in case, though not in your list
    }
    # Map and fill unknown with -1 or mode
    df_proc['education_encoded'] = df_proc['education_level'].map(edu_map).fillna(-1)
    
    # 3. Binary Encoding
    df_proc['gender_encoded'] = df_proc['gender'].map({'Male': 0, 'Female': 1})
    df_proc['marital_encoded'] = df_proc['marital_status'].map({'Single': 0, 'Married': 1})
    
    # 4. One-Hot Encoding for Nominal variables (Employment, Loan Purpose, Grade Letter)
    # We use pd.get_dummies
    cols_to_dummy = ['employment_status', 'loan_purpose', 'grade_letter']
    df_proc = pd.get_dummies(df_proc, columns=cols_to_dummy, drop_first=True)
    
    # 5. Feature Engineering (Ratios)
    # Interaction between income and loan amount is usually powerful
    df_proc['loan_to_income'] = df_proc['loan_amount'] / (df_proc['annual_income'] + 1)
    df_proc['monthly_debt'] = (df_proc['annual_income'] / 12) * df_proc['debt_to_income_ratio']
    
    # Drop original string columns that are now encoded
    cols_to_drop = ['grade_subgrade', 'education_level', 'gender', 'marital_status']
    df_proc = df_proc.drop(columns=cols_to_drop)
    
    return df_proc

# Apply preprocessing
df_processed = preprocess_data(df)

# Split back into Train and Test
train_final = df_processed[df_processed['is_train'] == 1].drop(columns=['is_train'])
test_final = df_processed[df_processed['is_train'] == 0].drop(columns=['is_train', 'loan_paid_back'])

# Define X and y
X = train_final.drop(columns=['loan_paid_back'])
y = train_final['loan_paid_back'].astype(int)
X_test = test_final.copy()

# Align columns (ensure test has same columns as train, fill missing with 0)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("Preprocessing complete.")


# XGBoost Parameters
xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    'tree_method': 'hist' # Faster training
}

# LightGBM Parameters
lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary',
    'metric': 'auc',
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}


N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Arrays to store predictions
oof_xgb = np.zeros(len(X))
oof_lgb = np.zeros(len(X))
test_pred_xgb = np.zeros(len(X_test))
test_pred_lgb = np.zeros(len(X_test))

print(f"Starting training with {N_FOLDS} folds...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Fold {fold + 1} ---")
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # --- XGBOOST ---
    model_xgb = xgb.XGBClassifier(**xgb_params)
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Predict (Probabilities for AUC)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    test_pred_xgb += model_xgb.predict_proba(X_test)[:, 1] / N_FOLDS
    
    print(f"XGB AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.5f}")
    
    # --- LIGHTGBM ---
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)[:, 1]
    test_pred_lgb += model_lgb.predict_proba(X_test)[:, 1] / N_FOLDS
    
    print(f"LGB AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.5f}")

print("\nTraining Finished.")


# Calculate Overall Scores for individual models
xgb_auc = roc_auc_score(y, oof_xgb)
lgb_auc = roc_auc_score(y, oof_lgb)

print(f"Overall XGBoost AUC: {xgb_auc:.5f}")
print(f"Overall LightGBM AUC: {lgb_auc:.5f}")

# --- ENSEMBLE (BLENDING) ---
# Simple Average (50/50 split)
oof_ensemble = (oof_xgb + oof_lgb) / 2
ensemble_auc = roc_auc_score(y, oof_ensemble)

print(f"Ensemble (50/50) AUC: {ensemble_auc:.5f}")

# Optional: Find optimal weights
# This iterates to find the best mix (e.g., 0.4*XGB + 0.6*LGB)
best_score = 0
best_weight = 0
for w in np.linspace(0, 1, 100):
    temp_oof = (w * oof_xgb) + ((1 - w) * oof_lgb)
    temp_score = roc_auc_score(y, temp_oof)
    if temp_score > best_score:
        best_score = temp_score
        best_weight = w

print(f"Best Weighted Ensemble AUC: {best_score:.5f} (Weight XGB: {best_weight:.2f})")


# Feature Importance Plot (XGBoost)
plt.figure(figsize=(10, 8))
xgb.plot_importance(model_xgb, max_num_features=20, height=0.5, title="XGBoost Feature Importance")
plt.show()


# Calculate final test predictions using the best weights found
final_test_pred = (best_weight * test_pred_xgb) + ((1 - best_weight) * test_pred_lgb)


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df.get('id', test_df.index), # Handle case if ID column is missing
    'loan_paid_back': final_test_pred
})

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully!")
print(submission.head())

