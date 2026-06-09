# ============================================================
# ğŸ�� STEP 1: Library Imports and Data Loading
# ============================================================

# Basic setup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")

# Quick structure overview
print("âœ… Data Loaded Successfully!\n")
print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")


print("\nğŸ“˜ Columns in train dataset:")
print(train.columns.tolist())


# Peek at data
print("\nğŸ”¹ Training Data Preview:")
display(train.head())


print("\nğŸ”¹ Test Data Preview:")
display(test.head())


# Data types & missing values
print("\nğŸ“Š Data Info:")
train.info()


print("\nğŸš« Missing Values in Train Set:")
print(train.isnull().sum())


print("\nğŸš« Missing Values in Test Set:")
print(test.isnull().sum())


# ============================================================
# ğŸ“Š STEP 2: Dataset Summary and Cleaning Check
# ============================================================

# Separate numerical and categorical columns
num_cols = train.select_dtypes(include=['int64', 'float64']).columns.drop('id')
cat_cols = train.select_dtypes(include='object').columns

print(f"Numerical Columns ({len(num_cols)}): {list(num_cols)}")
print(f"Categorical Columns ({len(cat_cols)}): {list(cat_cols)}")


# ------------------------------------------------------------
# Summary statistics for numerical features
# ------------------------------------------------------------
num_summary = train[num_cols].describe().T
num_summary['missing'] = train[num_cols].isnull().sum()
num_summary['unique'] = [train[c].nunique() for c in num_cols]
num_summary.style.background_gradient(cmap='Greens').format("{:.2f}")


# ------------------------------------------------------------
# Categorical summary
# ------------------------------------------------------------
cat_summary = pd.DataFrame({
    'unique_values': [train[c].nunique() for c in cat_cols],
    'most_frequent': [train[c].mode()[0] for c in cat_cols],
    'freq_count': [train[c].value_counts().iloc[0] for c in cat_cols]
}, index=cat_cols)

display(cat_summary.style.background_gradient(cmap='Purples'))


# ------------------------------------------------------------
# Target variable distribution
# ------------------------------------------------------------
plt.figure(figsize=(6,4))
ax = sns.countplot(data=train, x='loan_paid_back', palette='Set2')
plt.title("Target Variable Distribution â€” Loan Paid Back (1) vs Not (0)")
plt.xlabel("Loan Paid Back")
plt.ylabel("Count")
plt.bar_label(ax.containers[0], fmt='%d', label_type='edge')
plt.show()


# Display value counts
target_dist = train['loan_paid_back'].value_counts(normalize=True) * 100
print("\nğŸ�¯ Target Class Distribution (%):")
print(target_dist.round(2))


# ============================================================
# ğŸ“ˆ STEP 3A: Numerical Feature Distributions
# ============================================================

num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

plt.figure(figsize=(16, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[col], kde=True, bins=40, color='#1f77b4')
    plt.title(f"{col} Distribution", fontsize=12, weight='bold')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Skewness check
# ------------------------------------------------------------
skew_values = train[num_features].skew().sort_values(ascending=False)
print("ğŸ“Š Skewness of Numerical Features:")
print(skew_values)


# ============================================================
# ğŸ�¯ STEP 4A: Numerical Features vs Target
# ============================================================

num_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

plt.figure(figsize=(16, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=train, x='loan_paid_back', y=col, palette='Set2')
    plt.title(f"{col} vs Loan Paid Back", fontsize=12, weight='bold')
    plt.xlabel("Loan Paid Back (1 = Yes, 0 = No)")
    plt.ylabel(col)
plt.tight_layout()
plt.show()


# Correlation with target
corrs = train[num_features + ['loan_paid_back']].corr()['loan_paid_back'].sort_values(ascending=False)
print("ğŸ“ˆ Correlation of Numerical Features with Loan Paid Back:")
print(corrs)


# ============================================================
# ğŸ”¥ STEP 5A: Correlation Heatmap and Pairwise Relationships
# ============================================================

plt.figure(figsize=(8,6))
corr_matrix = train[['annual_income', 'debt_to_income_ratio', 'credit_score',
                     'loan_amount', 'interest_rate', 'loan_paid_back']].corr()

sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix", fontsize=13, weight='bold')
plt.show()


# Pairplot for key features vs target
key_features = ['credit_score', 'debt_to_income_ratio', 'interest_rate', 'loan_paid_back']
sns.pairplot(train[key_features], hue='loan_paid_back', palette='husl', diag_kind='kde', corner=True)
plt.suptitle("Pairwise Relationships among Key Features", y=1.02, fontsize=13, weight='bold')
plt.show()


!pip install -U scikit-learn imbalanced-learn


# ==========================================
# ğŸ“¦ Imports
# ==========================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

import optuna

import warnings
warnings.filterwarnings("ignore")


# ==========================================
# ğŸ”§ Prepare Features
# ==========================================
target_col = "loan_paid_back"

# Separate target and features
X = train.drop(columns=[target_col, "id"])
y = train[target_col].astype(float)

# Keep IDs for submission
test_ids = test["id"]
X_test = test.drop(columns=["id"])


# ============================================================
# âœ¨ FIX SKEWED FEATURES
# ============================================================

# Log transform annual_income (positive values only)
X["annual_income"] = np.log1p(X["annual_income"])
X_test["annual_income"] = np.log1p(X_test["annual_income"])

# Normalize debt_to_income_ratio using sqrt
X["debt_to_income_ratio"] = np.sqrt(X["debt_to_income_ratio"])
X_test["debt_to_income_ratio"] = np.sqrt(X_test["debt_to_income_ratio"])


# ============================================================
# ğŸ”¤ ENCODE CATEGORICAL FEATURES
# ============================================================
cat_cols = X.select_dtypes(include="object").columns.tolist()

encoder = OrdinalEncoder()
X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# Numerical columns (already clean)
num_cols = X.select_dtypes(exclude="object").columns.tolist()


# ============================================================
# âœ‚ï¸� TRAIN-VALIDATION SPLIT
# ============================================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ==========================================
# ğŸ�† Best Parameters from Optuna
# ==========================================
best_params = {
    'learning_rate': 0.046216594722431385, 
    'num_leaves': 81, 
    'max_depth': 4, 
    'min_child_samples': 141, 
    'subsample': 0.7706417252454062, 
    'colsample_bytree': 0.714224110342554, 
    'reg_alpha': 2.1924967649292544, 
    'reg_lambda': 2.057985586700453
}


# ==========================================
# ğŸš€ Train Final Model with Best Params
# ==========================================
best_params.update({
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "n_estimators": 3000,
    "random_state": 42
})

final_model = LGBMClassifier(**best_params)


# Fit with full resampled data
sm = SMOTE(sampling_strategy=0.5, random_state=42)
X_res, y_res = sm.fit_resample(X, y)

final_model.fit(X_res, y_res)


# ==========================================
# ğŸ“ˆ Generate Predictions
# ==========================================
probs_test = final_model.predict_proba(X_test)[:, 1]


# ==========================================
# ğŸ’¾ Save Submission
# ==========================================
submission = pd.DataFrame({
    "id": test_ids,
    "loan_paid_back": probs_test
})
submission.to_csv("submission.csv", index=False)

print("âœ… Submission file saved as: submission.csv")


submission

