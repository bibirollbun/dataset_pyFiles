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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")


# Step 1: Load the data
file_name_1 = '/kaggle/input/playground-series-s5e11/train.csv'
file_name_2 = '/kaggle/input/playground-series-s5e11/test.csv'
train = pd.read_csv(file_name_1) # index_col
test = pd.read_csv(file_name_2)
target = 'loan_paid_back'


numeric_cols = train.select_dtypes(include=['int64','float64']).columns.tolist()
numeric_cols.remove(target)

# Summary stats
display(train[numeric_cols].describe())

# Histograms + KDE
plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 2, i)
    sns.histplot(train[col], bins=40, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()


# Ratios and interaction features
train['loan_to_income'] = train['loan_amount'] / train['annual_income']
train['loan_dti_interaction'] = train['loan_amount'] * train['debt_to_income_ratio']
train['loan_interest_interaction'] = train['loan_amount'] * train['interest_rate']
train['credit_income_interaction'] = train['credit_score'] * train['annual_income']

# Log transforms
train['log_annual_income'] = np.log1p(train['annual_income'])
train['log_debt_to_income_ratio'] = np.log1p(train['debt_to_income_ratio'])
train['log_loan_to_income'] = np.log1p(train['loan_amount']) / np.log1p(train['annual_income'])


# Ratios and interaction features
test['loan_to_income'] = test['loan_amount'] / test['annual_income']
test['loan_dti_interaction'] = test['loan_amount'] * train['debt_to_income_ratio']
test['loan_interest_interaction'] = test['loan_amount'] * test['interest_rate']
test['credit_income_interaction'] = test['credit_score'] * test['annual_income']

# Log transforms
test['log_annual_income'] = np.log1p(test['annual_income'])
test['log_debt_to_income_ratio'] = np.log1p(test['debt_to_income_ratio'])
test['log_loan_to_income'] = np.log1p(test['loan_amount']) / np.log1p(test['annual_income'])


df_enc = train.copy()

# 1) Ordinal encode grade_subgrade (correct)
order = [
    'A1','A2','A3','A4','A5',
    'B1','B2','B3','B4','B5',
    'C1','C2','C3','C4','C5',
    'D1','D2','D3','D4','D5',
    'E1','E2','E3','E4','E5',
    'F1','F2','F3','F4','F5'
]
grade_map = {g: i+1 for i, g in enumerate(order)}
df_enc['grade_subgrade_ord'] = df_enc['grade_subgrade'].map(grade_map)

# Drop raw grade_subgrade (important!)
df_enc.drop(columns=['grade_subgrade'], inplace=True)


# 2) Group rare categories into "Other"
rare_thresh = 0.01
nominal_cols = ['gender', 'marital_status', 'education_level', 
                'employment_status', 'loan_purpose']

nominal_cols = [c for c in nominal_cols if c in df_enc.columns]

for col in nominal_cols:
    freqs = df_enc[col].value_counts(normalize=True)
    rare_values = freqs[freqs < rare_thresh].index.tolist()
    if rare_values:
        df_enc[col] = df_enc[col].replace(rare_values, 'Other')


# 3) One-hot encode nominal features
df_ohe = pd.get_dummies(df_enc[nominal_cols], prefix=nominal_cols, drop_first=True)
df_enc = pd.concat([df_enc.drop(columns=nominal_cols, axis=1), df_ohe], axis=1)

# 4) Optional frequency encoding helper (not used yet)
def add_frequency_encoding(df, col, new_col_name=None):
    if new_col_name is None:
        new_col_name = f"{col}_freq"
    freq = df[col].value_counts(normalize=True)
    df[new_col_name] = df[col].map(freq)
    return df


# Final checks
print("Final shape:", df_enc.shape)
display(df_enc.head())


# ------------------------------------------------------------
# STEP 5 — Train/Validation Split + Pipelines + Model Comparison
# Using all features (numeric + categorical)
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1) Separate features and target
# -------------------------------

X = df_enc.drop(columns=['id','loan_paid_back'], errors='ignore')
y = df_enc[target]

# -------------------------------
# 2) Train/validation split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 3) Identify numeric columns
# -------------------------------
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# -------------------------------
# 4) Preprocessing: scale numeric features
# -------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols)
    ],
    remainder="passthrough"  # keep categorical features
)

# # ------------------------------------------------
# # Compute class imbalance ratio
# # ------------------------------------------------
# neg, pos = y_train.value_counts()
# scale = neg / pos
# print("Class imbalance ratio (neg/pos):", scale)



# ============================================
# 6.1 Logistic Regression Hyperparameter Tuning
# ============================================

log_reg_model = LogisticRegression(
#    class_weight="balanced",
    max_iter=2000,
    n_jobs=-1
)

log_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", log_reg_model)
])

log_params = {
    "model__C": [0.1, 0.3, 1.0, 3.0, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"]
}

log_grid = GridSearchCV(
    log_pipe, log_params, cv=3,
    scoring="roc_auc", n_jobs=-1
)

log_grid.fit(X_train, y_train)

print("Best Logistic Regression AUC:", log_grid.best_score_)
log_grid.best_params_



# ============================================
# 6.2 XGBoost Hyperparameter Tuning
# ============================================

xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
#    scale_pos_weight=scale,   # <--- imbalance handling
    tree_method="hist",
    eval_metric="logloss"
)

xgb_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", xgb_model)
])

xgb_params = {
    "model__n_estimators": [200, 400],
    "model__learning_rate": [0.03, 0.05],
    "model__max_depth": [4, 6],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8]
}

xgb_grid = GridSearchCV(
    xgb_pipe, xgb_params, cv=3,
    scoring="roc_auc", n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

print("Best XGB AUC:", xgb_grid.best_score_)
xgb_grid.best_params_



# ============================================
# 7. Select Better Model (by ROC-AUC)
# ============================================

if xgb_grid.best_score_ > log_grid.best_score_:
    best_model = xgb_grid.best_estimator_
    print("Selected Model: XGBoost")
else:
    best_model = log_grid.best_estimator_
    print("Selected Model: Logistic Regression")


# ============================================
# 8. Validation Performance
# ============================================

y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)[:,1]

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1:", f1_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_proba))

print(classification_report(y_val, y_pred))



# Assume XGBoost performed best
final_model = best_model
final_model.fit(X, y)  # Train on full training set

# Prepare test data
df_test_enc = test.copy()

# Apply same feature engineering & encoding as training set
# (ordinal grade_subgrade, rare categories, one-hot encoding)
df_test_enc['grade_subgrade_ord'] = df_test_enc['grade_subgrade'].map(grade_map)

for col in nominal_cols:
    df_test_enc[col] = df_test_enc[col].replace(
        df_test_enc[col].value_counts(normalize=True)[lambda x: x<rare_thresh].index, 'Other'
    )

df_test_ohe = pd.get_dummies(df_test_enc[nominal_cols], drop_first=True)
df_test_enc = pd.concat([df_test_enc.drop(columns=nominal_cols), df_test_ohe], axis=1)

# Align columns with training set
df_test_enc = df_test_enc.reindex(columns=X.columns, fill_value=0)

# Predict probabilities
test_proba = final_model.predict_proba(df_test_enc)[:,1]

# Submission
submission = pd.DataFrame()
submission["id"] = test["id"]
submission["loan_paid_back"] = test_proba

submission.to_csv("submission.csv", index=False)
print("submission.csv created!")





