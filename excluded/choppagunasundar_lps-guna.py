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
# ============================================================
# LOAN REPAYMENT PREDICTION - ADVANCED PIPELINE
# Target: Maximize ROC AUC
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)




# Load datasets
train_df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("DATASET OVERVIEW")
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Target distribution:\n{train_df['loan_paid_back'].value_counts(normalize=True)}")

display(train_df.describe())



# ============================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    """
    Create sophisticated features for credit risk modeling
    """
    df = df.copy()
    
    # === FINANCIAL RISK RATIOS ===
    # Loan-to-Income ratio (higher = riskier)
    df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
    
    # Debt burden relative to income
    df['total_debt_burden'] = df['debt_to_income_ratio'] * df['annual_income']
    
    # Interest burden
    df['interest_burden'] = (df['loan_amount'] * df['interest_rate']) / 100
    
    # === CREDIT SCORE BINS ===
    # Create credit score bands (common in credit risk)
    df['credit_score_band'] = pd.cut(
        df['credit_score'], 
        bins=[0, 580, 640, 700, 760, 850],
        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
        include_lowest=True
    )
    
    # === LOAN CHARACTERISTICS ===
    # Extract grade and subgrade separately
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade_num'] = df['grade_subgrade'].str[1:].astype(int)
    
    # === INTERACTIONS ===
    # Credit score relative to interest rate
    df['credit_interaction'] = df['credit_score'] / df['interest_rate']
    
    # Income stability proxy
    df['income_stability_proxy'] = df['annual_income'] * df['debt_to_income_ratio']
    
    # === BINARY FLAGS ===
    # High-risk flags
    df['is_high_dti'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
    df['is_low_credit'] = (df['credit_score'] < 640).astype(int)
    df['is_large_loan'] = (df['loan_amount'] > df['loan_amount'].quantile(0.75)).astype(int)
    
    # === RISK SCORE (composite) ===
    # Create a weighted risk score
    df['risk_score'] = (
        (1 - df['credit_score']/850) * 0.4 +  # 40% weight on credit
        df['debt_to_income_ratio'] * 0.3 +    # 30% weight on DTI
        df['loan_to_income_ratio'] * 0.3       # 30% weight on LTI
    )
    
    return df

# Apply feature engineering
print("ğŸ”§ ENGINEERING FEATURES...")
train_df = engineer_features(train_df)
test_df = engineer_features(test_df)

print(f"New features created. Training shape: {train_df.shape}")
print(f"New features: {[col for col in train_df.columns if col not in ['id', 'loan_paid_back']]}")



# ============================================================
# 3. FEATURE DEFINITIONS
# ============================================================

# Separate target and IDs
TARGET = 'loan_paid_back'
ID_COL = 'id'

y = train_df[TARGET]
X = train_df.drop(columns=[TARGET, ID_COL])
X_test = test_df.drop(columns=[ID_COL])

# Define feature types
numeric_features = [
    'annual_income', 'debt_to_income_ratio', 'credit_score', 
    'loan_amount', 'interest_rate', 'loan_to_income_ratio',
    'total_debt_burden', 'interest_burden', 'credit_interaction',
    'income_stability_proxy', 'risk_score', 'subgrade_num'
]

categorical_features = [
    'gender', 'marital_status', 'education_level', 
    'employment_status', 'loan_purpose', 'grade_subgrade',
    'credit_score_band', 'grade'
]

binary_features = [
    'is_high_dti', 'is_low_credit', 'is_large_loan'
]

# Add binary features to appropriate list
numeric_features.extend(binary_features)

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")



# ============================================================
# 4. ADVANCED PREPROCESSING PIPELINE
# ============================================================

# Numeric transformer with robust imputation
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# Categorical transformer with frequency encoding for high cardinality
categorical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', LabelEncoder())  # Will use XGBoost's native handling
    ]
)

# Custom label encoder that handles unknown categories
class SafeLabelEncoder(LabelEncoder):
    def __init__(self):
        super().__init__()
        self.unknown_value = -1
        
    def fit(self, y):
        super().fit(y)
        return self
    
    def transform(self, y):
        """Transform, but assign -1 to unseen labels"""
        y = np.array(y)
        unseen_mask = ~np.isin(y, self.classes_)
        y_copy = y.copy()
        y_copy[unseen_mask] = self.classes_[0]  # Temporarily assign to known class
        result = super().transform(y_copy)
        result[unseen_mask] = self.unknown_value
        return result

# Apply preprocessing
print("ğŸ”„ PREPROCESSING DATA...")

# For XGBoost, we'll encode categoricals manually for better control
X_processed = X.copy()
X_test_processed = X_test.copy()

# Encode categorical features
encoders = {}
for col in categorical_features:
    le = SafeLabelEncoder()
    # Combine train and test to handle all categories
    combined = pd.concat([X_processed[col], X_test_processed[col]], axis=0)
    le.fit(combined.astype(str))
    
    X_processed[col] = le.transform(X_processed[col].astype(str))
    X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
    encoders[col] = le

print("âœ… Preprocessing complete")



# ============================================================
# 5. XGBOOST MODEL WITH OPTIMAL HYPERPARAMETERS
# ============================================================

# XGBoost parameters optimized for ROC AUC
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'tree_method': 'hist',
    'device': 'cpu',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

print("ğŸŒ² INITIALIZING XGBOOST MODEL...")
print(f"Parameters: {xgb_params}")

model = xgb.XGBClassifier(**xgb_params)



# ============================================================
# 6. STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================

print("ğŸ“ˆ PERFORMING STRATIFIED K-FOLD VALIDATION...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = []
fold_predictions = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_processed, y), 1):
    X_train_fold = X_processed.iloc[train_idx]
    X_valid_fold = X_processed.iloc[valid_idx]
    y_train_fold = y.iloc[train_idx]
    y_valid_fold = y.iloc[valid_idx]
    
    # Fit model with early stopping
    model_fold = xgb.XGBClassifier(**xgb_params)
    model_fold.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_valid_fold, y_valid_fold)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Predict and evaluate
    valid_pred_proba = model_fold.predict_proba(X_valid_fold)[:, 1]
    auc_score = roc_auc_score(y_valid_fold, valid_pred_proba)
    cv_scores.append(auc_score)
    
    print(f"Fold {fold}: ROC AUC = {auc_score:.4f}")
    
    # Store predictions for ensemble
    fold_predictions.append(model_fold.predict_proba(X_test_processed)[:, 1])

print(f"\nğŸ“Š CROSS-VALIDATION RESULTS:")
print(f"Mean ROC AUC: {np.mean(cv_scores):.4f} Â± {np.std(cv_scores):.4f}")
print(f"Individual folds: {cv_scores}")



# ============================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================

# Train final model on full data
print("ğŸ�¯ TRAINING FINAL MODEL ON FULL DATASET...")

final_model = xgb.XGBClassifier(**xgb_params)
final_model.fit(
    X_processed, y,
    eval_set=[(X_processed, y)],
    early_stopping_rounds=30,
    verbose=False
)

# Get feature importance
feature_importance = final_model.feature_importances_
feature_names = X_processed.columns

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nğŸ”� TOP 15 MOST IMPORTANT FEATURES:")
display(importance_df.head(15))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importances', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



# ============================================================
# 8. ENSEMBLE PREDICTIONS (OPTIONAL BUT RECOMMENDED)
# ============================================================

# Average predictions from all folds (more robust)
print("ğŸ�² GENERATING ENSEMBLE PREDICTIONS...")

# Mean of fold predictions
ensemble_proba = np.mean(fold_predictions, axis=0)

# Alternatively, use final model prediction
final_proba = final_model.predict_proba(X_test_processed)[:, 1]

# Weighted average (80% final model, 20% ensemble)
test_pred_proba = 0.8 * final_proba + 0.2 * ensemble_proba

print("âœ… Predictions generated")



# ============================================================
# 9. CREATE SUBMISSION FILE
# ============================================================

submission = pd.DataFrame({
    'id': test_df[ID_COL],
    'loan_paid_back': test_pred_proba
})

# Ensure probabilities are in [0,1] range
submission['loan_paid_back'] = submission['loan_paid_back'].clip(0, 1)

submission.to_csv("submission.csv", index=False)
print("ğŸ’¾ SUBMISSION FILE SAVED: submission.csv")
print(f"\nPreview:")
display(submission.head())



# ============================================================
# 10. VALIDATION CURVE VISUALIZATION
# ============================================================

# Plot ROC curve on validation set
print("ğŸ“Š GENERATING ROC CURVE...")

# Use last fold for visualization
fpr, tpr, _ = roc_curve(y_valid_fold, valid_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


