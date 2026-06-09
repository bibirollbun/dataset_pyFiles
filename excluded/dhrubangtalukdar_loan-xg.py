# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, heres several helpful packages to load

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


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')


df_test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


import numpy as np

def create_financial_features(df):
    # ğŸ’° Affordability / Income vs Loan
    df['income_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1)
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1)

    # ğŸ“‰ Debt Metrics
    df['total_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['available_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    df['debt_burden'] = df['debt_to_income_ratio'] * df['loan_amount']

    # ğŸ’³ Payment Ability
    df['monthly_payment'] = df['loan_amount'] * df['interest_rate'] / 1200
    df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12 + 1)
    df['affordability'] = df['available_income'] / (df['loan_amount'] + 1)

    # âš ï¸� Composite Risk
    df['default_risk'] = (
        0.40 * df['debt_to_income_ratio'] +
        0.35 * (850 - df['credit_score']) / 850 +
        0.25 * df['interest_rate'] / 100
    )

    # ğŸ§  Credit Behavior
    df['credit_utilization'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
    df['credit_interest_product'] = df['credit_score'] * df['interest_rate'] / 100

    # ğŸ“� Log Transforms
    df['annual_income_log'] = np.log1p(df['annual_income'])
    df['loan_amount_log'] = np.log1p(df['loan_amount'])

    # ğŸ�·ï¸� Loan Grade Features
    df['grade_letter'] = df['grade_subgrade'].str[0]
    df['grade_number'] = df['grade_subgrade'].str[1:].astype(int)

    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    df['grade_rank'] = df['grade_letter'].map(grade_map)

    return df



df_train = create_financial_features(df_train)
df_test  = create_financial_features(df_test)


categorical_cols = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose',
    'grade_subgrade',
    'grade_letter'
]

for col in categorical_cols:
    print(f"\nğŸ”¹ Column: {col}")

    train_vals = set(df_train[col].dropna().unique()) if col in df_train.columns else set()
    test_vals  = set(df_test[col].dropna().unique())  if col in df_test.columns else set()

    print(f"Train unique ({len(train_vals)}): {sorted(train_vals)}")
    print(f"Test  unique ({len(test_vals)}): {sorted(test_vals)}")

    # Extra: values only in one split
    print(f"Only in Train: {sorted(train_vals - test_vals)}")
    print(f"Only in Test : {sorted(test_vals - train_vals)}")


# Gender
gender_map = {
    'Male': 0,
    'Female': 1,
    'Other': 2
}

# Marital status
marital_map = {
    'Single': 0,
    'Married': 1,
    'Divorced': 2,
    'Widowed': 3
}

# Education level
education_map = {
    'High School': 0,
    "Bachelor's": 1,
    "Master's": 2,
    'PhD': 3,
    'Other': 4
}

# Employment status
employment_map = {
    'Employed': 0,
    'Self-employed': 1,
    'Unemployed': 2,
    'Student': 3,
    'Retired': 4
}

# Loan purpose
loan_purpose_map = {
    'Debt consolidation': 0,
    'Home': 1,
    'Car': 2,
    'Education': 3,
    'Medical': 4,
    'Business': 5,
    'Vacation': 6,
    'Other': 7
}

# Grade letter (ordinal)
grade_letter_map = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6
}


for df in [df_train, df_test]:
    df['gender'] = df['gender'].map(gender_map)
    df['marital_status'] = df['marital_status'].map(marital_map)
    df['education_level'] = df['education_level'].map(education_map)
    df['employment_status'] = df['employment_status'].map(employment_map)
    df['loan_purpose'] = df['loan_purpose'].map(loan_purpose_map)
    df['grade_letter'] = df['grade_letter'].map(grade_letter_map)


grade_subgrade_map = {
    f"{l}{n}": (i * 5 + n)
    for i, l in enumerate(['A','B','C','D','E','F'])
    for n in range(1, 6)
}

for df in [df_train, df_test]:
    df['grade_subgrade'] = df['grade_subgrade'].map(grade_subgrade_map)



df_train


df_train.columns


train_df = df_train.copy()


# ===============================
# XGBOOST PIPELINE (WITH TQDM)
# ===============================

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# -------------------------------
# 1. PREPARE DATA
# -------------------------------
TARGET = 'loan_paid_back'

X = train_df.drop(columns=[TARGET, 'id'])
y = train_df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------------------------------
# 2. DROP REDUNDANT FEATURES
# -------------------------------
drop_cols = ['id', 'annual_income', 'loan_amount', 'total_debt']
drop_cols = [c for c in drop_cols if c in X_train.columns]

X_train = X_train.drop(columns=drop_cols)
X_val   = X_val.drop(columns=drop_cols)

# -------------------------------
# 3. XGBOOST MODEL
# -------------------------------
xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost model...")
xgb.fit(X_train, y_train)

# -------------------------------
# 4. THRESHOLD TUNING (TQDM)
# -------------------------------
probs = xgb.predict_proba(X_val)[:, 1]

best_t, best_f1 = 0.5, 0

for t in tqdm(np.arange(0.2, 0.8, 0.02), desc="Threshold tuning"):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_val, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"\nBest threshold: {best_t:.2f}")

# -------------------------------
# 5. FINAL EVALUATION
# -------------------------------
final_preds = (probs >= best_t).astype(int)

print("\nFinal Metrics:")
print("Accuracy :", accuracy_score(y_val, final_preds))
print("Precision:", precision_score(y_val, final_preds))
print("Recall   :", recall_score(y_val, final_preds))
print("F1 Score :", f1_score(y_val, final_preds))

# -------------------------------
# 6. FEATURE IMPORTANCE
# -------------------------------
feature_importance = pd.Series(
    xgb.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 15 Important Features:")
for f, v in feature_importance.head(15).items():
    print(f"{f:<30} {v:.4f}")



# ===============================
# XGBOOST FULL TRAIN + TEST PRED
# ===============================

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# -------------------------------
# 1. PREPARE DATA
# -------------------------------
TARGET = 'loan_paid_back'

X_train_full = train_df.drop(columns=[TARGET, 'id'])
y_train_full = train_df[TARGET]

X_test = df_test.drop(columns=['id'])

# -------------------------------
# 2. DROP REDUNDANT FEATURES
# -------------------------------
drop_cols = ['annual_income', 'loan_amount', 'total_debt']
drop_cols = [c for c in drop_cols if c in X_train_full.columns]

X_train_full = X_train_full.drop(columns=drop_cols)
X_test       = X_test.drop(columns=drop_cols)

# -------------------------------
# 3. XGBOOST MODEL (FINAL)
# -------------------------------
xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    min_child_weight=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=5.0,
    scale_pos_weight=(y_train_full == 0).sum() / (y_train_full == 1).sum(),
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost on full train_df...")
xgb.fit(X_train_full, y_train_full)

# -------------------------------
# 4. PREDICT ON TEST
# -------------------------------
test_probs = xgb.predict_proba(X_test)[:, 1]

# Use tuned threshold (change if needed)
BEST_THRESHOLD = 0.45
test_preds = (test_probs >= BEST_THRESHOLD).astype(int)

# -------------------------------
# 5. CREATE SUBMISSION
# -------------------------------
submission = pd.DataFrame({
    'id': df_test['id'],
    'loan_paid_back': test_preds
})

submission.to_csv('submission.csv', index=False)

print("âœ… submission.csv saved successfully")
print(submission.head())





