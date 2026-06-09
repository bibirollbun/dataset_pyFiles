

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import time

train = pd.read_csv('/kaggle/input/data2google/train(1).csv')
test = pd.read_csv('/kaggle/input/data2google/test(1).csv')

X = train.drop(['id','loan_paid_back'], axis=1)
y = train['loan_paid_back']
test_ids = test['id']
test_X = test.drop(['id'], axis=1)

cat_cols = ['gender', 'marital_status', 'education_level', 
            'employment_status', 'loan_purpose', 'grade_subgrade']

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna('Unknown')
        test_X[col] = test_X[col].fillna('Unknown')
    else:
        X[col] = X[col].fillna(X[col].median())
        test_X[col] = test_X[col].fillna(X[col].median())

for col in cat_cols:
    le = LabelEncoder()
    le.fit(list(X[col].values) + list(test_X[col].values))
    X[col] = le.transform(X[col])
    test_X[col] = le.transform(test_X[col])

numeric_cols = X.select_dtypes(include=np.number).columns

for col in numeric_cols:
    X[f'{col}_log'] = np.log1p(X[col])
    test_X[f'{col}_log'] = np.log1p(test_X[col])

if 'loan_amount' in numeric_cols and 'annual_income' in numeric_cols:
    X['loan_income_ratio'] = X['loan_amount'] / (X['annual_income'] + 1)
    test_X['loan_income_ratio'] = test_X['loan_amount'] / (test_X['annual_income'] + 1)

if 'loan_amount' in numeric_cols and 'interest_rate' in numeric_cols:
    X['loan_int_ratio'] = X['loan_amount'] * X['interest_rate']
    test_X['loan_int_ratio'] = test_X['loan_amount'] * test_X['interest_rate']

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
test_X[numeric_cols] = scaler.transform(test_X[numeric_cols])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=8,
    colsample_bytree=0.8,
    subsample=0.85,
    min_child_samples=30,
    random_state=42
)

lgb_model.fit(X_train, y_train)

val_preds = lgb_model.predict_proba(X_val)[:,1]
roc_score = roc_auc_score(y_val, val_preds)
print("Validation ROC-AUC:", roc_score)

lgb_model.fit(X, y)

test_preds = lgb_model.predict_proba(test_X)[:,1]

submission = pd.DataFrame({'id': test_ids, 'loan_paid_back': test_preds})
submission_file = '/kaggle/working/submission.csv'
submission.to_csv(submission_file, index=False)
print(f"✅ Submission file created: {submission_file}")


