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


train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")

test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

print("\n-----------------train-------------------\n")
display(train)
print("\n-------------------test-------------------\n")
display(test)


test_id=test["id"]


print(train.shape)
print(test.shape)

print("\n---------train-----------\n")
print(train.info())
print("\n---------test-----------\n")
print(test.info())

print("\n----------train---------\n")
print(train.describe())
print("\n-------test----------\n")
print(test.describe())

# %% [code]
print("\n---------train---------\n")
print(train.dtypes)
print("\n---------test---------\n")
print(test.dtypes)

# # Missing Value Check

print("\ntrain\n")
print(train.isnull().sum())
print("\ntest\n")
print(test.isnull().sum())


train


print("\n Raw data \n")
print(train["loan_paid_back"].value_counts())
print("\n percentage \n")
print(train["loan_paid_back"].value_counts(normalize=True))

sns.countplot(x="loan_paid_back",data=train)
plt.title("Target Distribution ")
plt.xlabel("loan_paid_back")
plt.ylabel("count")


num_cols = ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
cat_cols = ["gender","marital_status","education_level","employment_status","loan_purpose","grade_subgrade"]


for col in cat_cols:
    plt.figure(figsize=(8, 5))
    sns.countplot(x=col, data=train)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()


for col in num_cols:
 plt.figure(figsize=(7,4))
 sns.histplot(train[col],bins=30,kde=True)
 plt.title(f"measurment of {col}")
 plt.xlabel(f"{col}")
 plt.ylabel("frequency")
 plt.tight_layout()


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# =========================================================
# ğŸ§® 1. Basic Derived Columns
# =========================================================

train['monthly_income'] = train['annual_income'] / 12
train['loan_to_income_ratio'] = train['loan_amount'] / train['annual_income']
train['loan_to_income_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

# =========================================================
# ğŸ§  2. Credit and Interest Categorization
# =========================================================

train['credit_category'] = pd.cut(
    train['credit_score'],
    bins=[0, 580, 670, 740, 800, 900],
    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
)

train['interest_category'] = pd.cut(
    train['interest_rate'],
    bins=[0, 8, 12, 20, 50],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# =========================================================
# ğŸ‘¨â€�ğŸ‘©â€�ğŸ‘§â€�ğŸ‘¦ 3. Gender & Marital Encoding
# =========================================================

train['gender_encoded'] = train['gender'].map({'Male': 1, 'Female': 0}).fillna(-1)
train['is_married'] = train['marital_status'].apply(
    lambda x: 1 if str(x).lower() == 'married' else 0
)

# =========================================================
# ğŸ�“ 4. Employment & Loan Purpose Encoding
# =========================================================

le_emp = LabelEncoder()
le_purpose = LabelEncoder()

train['employment_status'] = train['employment_status'].astype(str)
train['loan_purpose'] = train['loan_purpose'].astype(str)

train['employment_encoded'] = le_emp.fit_transform(train['employment_status'])
train['loan_purpose_encoded'] = le_purpose.fit_transform(train['loan_purpose'])

# =========================================================
# ğŸ§¾ 5. Grade Conversion (e.g., A1 â†’ 1, B3 â†’ 8)
# =========================================================

def grade_to_numeric(grade):
    if pd.isna(grade):
        return np.nan
    letter = grade[0].upper()
    number = int(grade[1]) if grade[1:].isdigit() else 0
    letter_val = ord(letter) - ord('A')
    return letter_val * 5 + number

train['grade_numeric'] = train['grade_subgrade'].apply(grade_to_numeric)

# =========================================================
# ğŸ’° 6. Financial Stability Index
# =========================================================

train['financial_stability_index'] = (
    (train['credit_score'] / 850) * (1 - train['debt_to_income_ratio'])
).clip(lower=0)

# =========================================================
# ğŸ§© 7. Optional Categoricals (Income & DTI Categories)
# =========================================================

train['income_category'] = pd.cut(
    train['annual_income'],
    bins=[0, 300000, 800000, 1500000, float('inf')],
    labels=['Low', 'Medium', 'High', 'Very High']
)

train['debt_to_income_category'] = pd.cut(
    train['debt_to_income_ratio'],
    bins=[0, 0.2, 0.4, 1],
    labels=['Low', 'Medium', 'High']
)

# =========================================================
# âœ… Final Columns (Clean View)
# =========================================================

optimized_columns = [
    'id',
    'annual_income',
    'monthly_income',
    'debt_to_income_ratio',
    'loan_amount',
    'loan_to_income_ratio',
    'credit_score',
    'credit_category',
    'interest_rate',
    'interest_category',
    'gender_encoded',
    'is_married',
    'education_level',
    'employment_status',
    'employment_encoded',
    'loan_purpose',
    'loan_purpose_encoded',
    'grade_subgrade',
    'grade_numeric',
    'financial_stability_index',
    'income_category',
    'debt_to_income_category',
    'loan_paid_back'
]

train = train[optimized_columns]




# Preview few rows
train.head()



# =========================================================
# ğŸ“Š Loan Dataset Feature Engineering â€“ Kaggle Notebook
# =========================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# âœ… Ensure your dataset is already loaded as df
# Example:
# df = pd.read_csv("/kaggle/input/loan-data/loan_data.csv")

# =========================================================
# ğŸ§® 1. Basic Derived Columns
# =========================================================

test['monthly_income'] = test['annual_income'] / 12
test['loan_to_income_ratio'] = test['loan_amount'] / test['annual_income']
test['loan_to_income_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

# =========================================================
# ğŸ§  2. Credit and Interest Categorization
# =========================================================

test['credit_category'] = pd.cut(
    test['credit_score'],
    bins=[0, 580, 670, 740, 800, 900],
    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
)

test['interest_category'] = pd.cut(
    test['interest_rate'],
    bins=[0, 8, 12, 20, 50],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# =========================================================
# ğŸ‘¨â€�ğŸ‘©â€�ğŸ‘§â€�ğŸ‘¦ 3. Gender & Marital Encoding
# =========================================================

test['gender_encoded'] = test['gender'].map({'Male': 1, 'Female': 0}).fillna(-1)
test['is_married'] = test['marital_status'].apply(
    lambda x: 1 if str(x).lower() == 'married' else 0
)

# =========================================================
# ğŸ�“ 4. Employment & Loan Purpose Encoding
# =========================================================

le_emp = LabelEncoder()
le_purpose = LabelEncoder()

test['employment_status'] = test['employment_status'].astype(str)
test['loan_purpose'] = test['loan_purpose'].astype(str)

test['employment_encoded'] = le_emp.fit_transform(test['employment_status'])
test['loan_purpose_encoded'] = le_purpose.fit_transform(test['loan_purpose'])

# =========================================================
# ğŸ§¾ 5. Grade Conversion (e.g., A1 â†’ 1, B3 â†’ 8)
# =========================================================

def grade_to_numeric(grade):
    if pd.isna(grade):
        return np.nan
    letter = grade[0].upper()
    number = int(grade[1]) if grade[1:].isdigit() else 0
    letter_val = ord(letter) - ord('A')
    return letter_val * 5 + number

test['grade_numeric'] = test['grade_subgrade'].apply(grade_to_numeric)

# =========================================================
# ğŸ’° 6. Financial Stability Index
# =========================================================

test['financial_stability_index'] = (
    (test['credit_score'] / 850) * (1 - test['debt_to_income_ratio'])
).clip(lower=0)

# =========================================================
# ğŸ§© 7. Optional Categoricals (Income & DTI Categories)
# =========================================================

test['income_category'] = pd.cut(
    test['annual_income'],
    bins=[0, 300000, 800000, 1500000, float('inf')],
    labels=['Low', 'Medium', 'High', 'Very High']
)

test['debt_to_income_category'] = pd.cut(
    test['debt_to_income_ratio'],
    bins=[0, 0.2, 0.4, 1],
    labels=['Low', 'Medium', 'High']
)

# =========================================================
# âœ… Final Columns (Clean View)
# =========================================================

optimized_columns = [
    'id',
    'annual_income',
    'monthly_income',
    'debt_to_income_ratio',
    'loan_amount',
    'loan_to_income_ratio',
    'credit_score',
    'credit_category',
    'interest_rate',
    'interest_category',
    'gender_encoded',
    'is_married',
    'education_level',
    'employment_status',
    'employment_encoded',
    'loan_purpose',
    'loan_purpose_encoded',
    'grade_subgrade',
    'grade_numeric',
    'financial_stability_index',
    'income_category',
    'debt_to_income_category'
]

test = test[optimized_columns]




# Preview few rows
test.head()



cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = train.select_dtypes(include=["int64","float64"]).columns.tolist()

print("Categorical:", cat_cols)
print("Numeric:", num_cols)


#train data
encoder = OneHotEncoder(sparse=False, drop="first")
encoded = encoder.fit_transform(train[cat_cols])
encoded_cols = encoder.get_feature_names_out(cat_cols)
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=train.index)
train_encoded = train.drop(columns=cat_cols).join(encoded_df)
display(train_encoded.head())


cat_cols = test.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = test.select_dtypes(include=["int64","float64"]).columns.tolist()

print("Categorical:", cat_cols)
print("Numeric:", num_cols)


#test data
encoder = OneHotEncoder(sparse=False, drop="first")
encoded = encoder.fit_transform(test[cat_cols])
encoded_cols = encoder.get_feature_names_out(cat_cols)
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=test.index)
test_encoded = test.drop(columns=cat_cols).join(encoded_df)
display(test_encoded)


#col=["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
#train_encoded[col] = np.log1p(train_encoded[col])
    #test_encoded[col] = np.log1p(test_encoded[col])


from sklearn.preprocessing import StandardScaler

#train
# Scale numerical features
numerical_cols =  ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
scaler = StandardScaler()
train_encoded[numerical_cols] = scaler.fit_transform(train_encoded[numerical_cols])


#train
# Scale numerical features
#scaler = StandardScaler()
#train_encoded = scaler.fit_transform(train_encoded)


#print(train_encoded)


#test
# Scale numerical features
numerical_cols =  ["annual_income","debt_to_income_ratio","credit_score","loan_amount","interest_rate"]
scaler = StandardScaler()
test_encoded[numerical_cols] = scaler.fit_transform(test_encoded[numerical_cols])


from  sklearn.model_selection import train_test_split

X = train_encoded.drop(['id', 'loan_paid_back'], axis=1)
y = train_encoded['loan_paid_back']

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)


#X_test = test_encoded.drop(['id'], axis=1)
X_test = test_encoded


from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Use class weights from earlier
#class_weights = [weights[0], weights[1]]

cat_model = CatBoostClassifier(iterations=2000,learning_rate=0.05,depth=8,l2_leaf_reg=5,random_seed=42,eval_metric="AUC",verbose=200)
#class_weights=class_weights,
# Train
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

# Predictions (probabilities for AUC)
cat_pred_proba = cat_model.predict_proba(X_val)[:, 1]
cat_pred = cat_model.predict(X_val)

# Metrics
print("ROC AUC Score (CatBoost):", roc_auc_score(y_val, cat_pred_proba))
print("F1 Score (CatBoost):", f1_score(y_val, cat_pred))


pred = cat_model.predict(X_test)
predict_df = pd.DataFrame(pred, columns=['loan_paid_back'])
submission = pd.concat([test['id'], predict_df], axis=1)

display(submission.head())
print(submission.isnull().sum())


submission.to_csv('submission1.csv', index=False)
print("Submission file saved as 'submission.csv'")


# =============================
# ğŸ”� XGBoost + GridSearchCV Optimization
# =============================

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


# Split your dataset into features and target
# (If already done earlier, skip this)
# X = train_df.drop(['target_column', 'id'], axis=1)
# y = train_df['target_column']

# =============================
# ğŸ�¯ Base Model
# =============================
xgb_base = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)


# =============================
# ğŸ§© Parameter Grid for Optimization
# =============================
param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3]
}

# =============================
# ğŸ§  Grid Search Setup
# =============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    n_jobs=-1
)

# =============================
# âš¡ Fit GridSearchCV
# =============================
grid_search.fit(X_train, y_train)

print("âœ… Best Parameters:", grid_search.best_params_)
print("ğŸ�† Best ROC-AUC (CV):", grid_search.best_score_)

# =============================
# ğŸš€ Train Final Model with Best Params
# =============================
best_xgb = grid_search.best_estimator_
best_xgb.fit(X_train, y_train)


# =============================
# ğŸ“ˆ ROC-AUC Evaluation
# =============================
y_pred_proba = best_xgb.predict_proba(X_valid)[:, 1]
roc_auc = roc_auc_score(y_valid, y_pred_proba)
print(f"âœ… Validation ROC-AUC: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_valid, y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve - XGBoost Classifier (Optimized)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()


# =============================
# ğŸ§© Prediction for Test Set
# =============================
# Replace X_test and test_ids with your dataset variables
# X_test = test_df.drop(['id'], axis=1)
# test_ids = test_df['id']

test_preds = best_xgb.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'id': test_ids,
    'loan_paid_back': test_preds  # or your actual target name
})

submission.to_csv('submission.csv', index=False)
print("ğŸ“� Submission file saved as submission.csv")
submission.head()




