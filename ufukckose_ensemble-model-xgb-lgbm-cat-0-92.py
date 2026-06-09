# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load datasets
df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


# Initial Exploration
df.head()


# Select all numeric columns
num_cols = df.select_dtypes("number").columns
# Drop the ID column and the target variable 'loan_paid_back'
num_cols = num_cols.drop(["id", "loan_paid_back"])
print("Numeric columns selected for histograms:")
print(num_cols)
print("\n")


# Data Visualization: 
# Numeric Histograms
df[num_cols].hist(bins = 20 ,figsize = (15, 15), color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.suptitle("Histograms of the Numeric Columns", fontsize=30, fontweight="bold")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# Target (loan_paid_back) distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="loan_paid_back", color="mediumorchid", edgecolor="purple", linewidth=1.2)
plt.title("Distribution of Loan Paid Back (Target)", fontweight = "bold" , fontsize=30)
plt.xlabel("Loan Paid Back (1 = Yes, 0 = No)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))

# Compute correlation matrix 
# this includes all numeric cols, including the target
corr = df.corr(numeric_only=True)

# Plot heatmap
sns.heatmap(corr, annot=True, cmap="Purples", fmt=".2f", annot_kws={"size": 10})
plt.title("Correlation Matrix", fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# Mean loan_paid_back by education and employment
# ------------------------------
# Create the pivot table
pivot_ee = df.pivot_table(
    values="loan_paid_back",
    index="education_level",
    columns="employment_status",
    aggfunc="mean"
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_ee, annot=True, fmt=".2f", cmap="Purples", linewidths=.5)
plt.title("Mean Loan Paid Back by Education Level & Employment Status", fontsize=16)
plt.xlabel("Employment Status")
plt.ylabel("Education Level")
plt.tight_layout()
plt.show()


# Feature Engineering
SAFE_DIV = 1e-6

def ftr_eng_loans(X):
    # Copy input
    df = X.copy()

    # Ordinal and grade features
    df['grade'] = df['grade_subgrade'].str[0]
    grade_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    df['grade_numeric'] = df['grade'].map(grade_map).fillna(0)
    df['subgrade_numeric'] = df['grade_subgrade'].str[1].astype(int).fillna(0)
    df['full_grade_score'] = (df['grade_numeric'] * 5) + df['subgrade_numeric']

    # We treat 'Other' as 1 (similar risk profile to high school in most credit data)
    edu_map = {
        'High School': 1,
        'Other': 1,
        "Bachelor's": 2,
        "Master's": 3,
        'PhD': 4
    }
    df['education_numeric'] = df['education_level'].map(edu_map).fillna(1)

    
    # ratio and financial Health Features
    df['loan_to_income_ratio'] = df['loan_amount'] / (df['annual_income'] + SAFE_DIV)
    df['annual_debt'] = df['annual_income'] * df['debt_to_income_ratio']
    df['monthly_income'] = df['annual_income'] / 12
    df['monthly_interest_proxy'] = (df['loan_amount'] * (df['interest_rate'] / 100)) / 12
    df['payment_to_monthly_income_ratio'] = df['monthly_interest_proxy'] / (df['monthly_income'] + SAFE_DIV)
    df['disposable_income'] = df['annual_income'] - df['annual_debt']
    df['loan_to_disposable_income'] = df['loan_amount'] / (df['disposable_income'] + SAFE_DIV)
    df['total_debt_with_loan'] = df['annual_debt'] + df['loan_amount']
    df['new_debt_to_income_ratio'] = df['total_debt_with_loan'] / (df['annual_income'] + SAFE_DIV)

    # Transforms
    df['log_annual_income'] = np.log1p(df['annual_income'])
    df['log_loan_amount'] = np.log1p(df['loan_amount'])
    df['log_interest_rate'] = np.log1p(df['interest_rate'])
    df['log_credit_score'] = np.log1p(df['credit_score'])
    
    df['credit_score_sq'] = df['credit_score'] ** 2
    df['dti_sq'] = df['debt_to_income_ratio'] ** 2

    # Interactions
    df['loan_x_interest'] = df['loan_amount'] * df['interest_rate']
    df['income_x_score'] = df['annual_income'] * df['credit_score']
    df['loan_per_score'] = df['loan_amount'] / (df['credit_score'] + SAFE_DIV)
    df['credit_x_dti'] = df['credit_score'] * df['debt_to_income_ratio']
    df['log_income_x_score'] = df['log_annual_income'] * df['log_credit_score']
    
    # Risk
    df['credit_score_normalized'] = df['credit_score'] / 850.0
    df['credit_risk_score'] = 1 - df['credit_score_normalized']

    # Freq encoding
    freq_cols = ['grade_subgrade', 'loan_purpose', 'employment_status']

    for col in freq_cols:
        if col in df.columns:
            # normalize=True creates a percentage (0 to 1)
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col + '_freq'] = df[col].map(freq_map)
            df[col + '_freq'] = df[col + '_freq'].fillna(0)
            
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    return df

print("Applying Feature Engineering...")
df_eng = ftr_eng_loans(df)
test_eng = ftr_eng_loans(test)


# Define features and target
X = df_eng.drop(columns=['loan_paid_back'])
y = df_eng['loan_paid_back']
X_test = test_eng.copy()


# Drop the original string 'education_level'
# We mapped this to 'education_numeric' (1,2,3,4).
cols_to_drop = ['education_level']
X = X.drop(columns=cols_to_drop, errors='ignore')
X_test = X_test.drop(columns=cols_to_drop, errors='ignore')

# Identify columns automatically
# 'education_numeric' is now a number so it correctly falls into num_cols
num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Total Numerical Columns: {len(num_cols)}")
print(f"Total Categorical Columns: {len(cat_cols)}")
print(f"Categorical Columns targeted: {cat_cols}")


# Preprocessing Pipeline

# RobustScaler deals better with income outliers
numeric_transformer = RobustScaler()

# For xgb, lgbm, rf (needs OneHot)
preprocessor_standard = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='passthrough'
)

# For CatBoost
preprocessor_cat = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ], remainder='passthrough'
)


xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'tree_method': 'hist',
    'device': "cuda",
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_jobs': -1
}


lgbm_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

cat_params = {
    'iterations': 2000,
    'learning_rate': 0.01,
    'depth': 6,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 0,
    'random_seed': 42,
    'allow_writing_files': False
}

rf_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'min_samples_split': 10,
    'n_jobs': -1,
    'random_state': 42
}


# Create Pipelines
pipe_xgb = Pipeline([('preprocessor', preprocessor_standard), ('model', XGBClassifier(**xgb_params))])
pipe_lgbm = Pipeline([('preprocessor', preprocessor_standard), ('model', LGBMClassifier(**lgbm_params))])
pipe_rf = Pipeline([('preprocessor', preprocessor_standard), ('model', RandomForestClassifier(**rf_params))])

pipe_cat = Pipeline([('preprocessor', preprocessor_cat), ('model', CatBoostClassifier(**cat_params))])


# Stacking clf
estimators = [
    ('xgb', pipe_xgb),
    ('lgbm', pipe_lgbm),
    ('cat', pipe_cat),
    ('rf', pipe_rf)
]

# Log reg is the meta
meta_model = LogisticRegression(random_state=42, C=0.1, solver='liblinear')

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',
    n_jobs=1,
    passthrough=False 
)


# Train and pred
stack.fit(X, y)
test_preds_proba = stack.predict_proba(X_test)[:, 1]


submission = pd.DataFrame({
    "id": test["id"],
    "loan_paid_back": test_preds_proba
})

filename = "submission_stacking_tuned_TRIO.csv"
submission.to_csv(filename, index=False)

