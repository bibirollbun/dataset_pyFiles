# --------------------------------------------
#   CREDIT DEFAULT PREDICTION - FULL EDA
#   Works directly in Kaggle Notebook
# --------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Settings
plt.style.use('default')
sns.set_palette("viridis")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# --------------------------------------------
# 1. LOAD DATA
# --------------------------------------------

df = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")

print("Shape:", df.shape)
df.head()



# --------------------------------------------
# 2. BASIC OVERVIEW
# --------------------------------------------

print("\n--- INFO ---")
df.info()

print("\n--- DESCRIPTIVE STATS ---")
df.describe().T



# --------------------------------------------
# 3. CHECK TARGET DISTRIBUTION
# --------------------------------------------

target = "SeriousDlqin2yrs"

plt.figure(figsize=(6,4))
sns.countplot(data=df, x=target)
plt.title("Target Distribution (Default vs Non-default)")
plt.show()

df[target].value_counts(normalize=True)



# --------------------------------------------
# 4. CHECK MISSING VALUES
# --------------------------------------------

missing = df.isnull().mean().sort_values(ascending=False)

plt.figure(figsize=(10,6))
missing.plot(kind="bar")
plt.title("Missing Value Percentage by Feature")
plt.ylabel("Missing %")
plt.show()

missing



# --------------------------------------------
# 5. UNIVARIATE ANALYSIS (Numerical Features)
# --------------------------------------------

num_cols = df.select_dtypes(include=['int64','float64']).columns
num_cols = num_cols.drop(target)

for col in num_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()



# --------------------------------------------
# 6. OUTLIER ANALYSIS USING BOXPLOTS
# --------------------------------------------

for col in num_cols:
    plt.figure(figsize=(7,3))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.show()



# --------------------------------------------
# 7. CORRELATION ANALYSIS
# --------------------------------------------

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



# --------------------------------------------
# 8. CORRELATION WITH TARGET
# --------------------------------------------

corr_target = df.corr()[target].sort_values(ascending=False)
print(corr_target)

plt.figure(figsize=(8,5))
corr_target.drop(target).plot(kind='bar')
plt.title("Correlation of Features with SeriousDlqin2yrs")
plt.show()



# --------------------------------------------
# 9. BIVARIATE ANALYSIS – TARGET vs FEATURES
# --------------------------------------------

for col in num_cols:
    plt.figure(figsize=(7,4))
    sns.boxplot(x=target, y=col, data=df)
    plt.title(f"{col} vs {target}")
    plt.show()



# --------------------------------------------
# 10. FEATURE-SPECIFIC EDA
# --------------------------------------------

# a) Monthly income distribution
plt.figure(figsize=(8,5))
sns.histplot(df['MonthlyIncome'], bins=40, kde=True)
plt.title("Monthly Income Distribution")
plt.show()

# b) Debt ratio distribution
plt.figure(figsize=(8,5))
sns.histplot(df['DebtRatio'], bins=40, kde=True)
plt.title("Debt Ratio Distribution")
plt.xlim(0,5)
plt.show()

# c) Number of Dependents
plt.figure(figsize=(6,4))
sns.countplot(x='NumberOfDependents', data=df)
plt.title("Number of Dependents Distribution")
plt.show()

# d) Revolving utilization
plt.figure(figsize=(8,5))
sns.histplot(df['RevolvingUtilizationOfUnsecuredLines'], bins=50, kde=True)
plt.title("Revolving Utilization Distribution")
plt.xlim(0,3)
plt.show()



# --------------------------------------------
# 11. MULTICOLLINEARITY CHECK (PAIR PLOTS)
# --------------------------------------------

sample_df = df.sample(3000, random_state=42)

sns.pairplot(sample_df[['RevolvingUtilizationOfUnsecuredLines',
                        'age',
                        'DebtRatio',
                        'MonthlyIncome',
                        target]], diag_kind="kde")
plt.show()



# --------------------------------------------
# 12. SUMMARY OF KEY FINDINGS
# --------------------------------------------

print("""
SUMMARY OF EDA FINDINGS
------------------------

1. TARGET IMBALANCE:
   -> ~6-7% defaults (highly imbalanced)

2. MISSING VALUES:
   -> MonthlyIncome has ~20% missing
   -> NumberOfDependents also missing

3. OUTLIERS:
   -> RevolvingUtilizationOfUnsecuredLines has values > 1 (very high)
   -> DebtRatio extreme values exist

4. STRONGEST CORRELATIONS:
   -> NumberOfTimes90DaysLate
   -> NumberOfTime60-89DaysPastDueNotWorse
   -> NumberOfTime30-59DaysPastDueNotWorse
   (Past-due counts are strongest predictors)

5. AGE:
   -> Older people default less

6. DEBT RATIO:
   -> Higher values associated with higher risk

7. INCOME:
   -> Lower income groups default more

8. VARIANCE / SKEWNESS:
   -> Many variables extremely skewed => log-transform may help
""")



# -------------------------------------------------------
# LOGISTIC REGRESSION MODEL FOR CREDIT DEFAULT PREDICTION
# -------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    classification_report, 
    roc_curve
)

# Load data
df = pd.read_csv("/kaggle/input/GiveMeSomeCredit/cs-training.csv")

# Drop ID column if present
df = df.rename(columns={"Unnamed: 0": "ID"})
if "ID" in df.columns:
    df = df.drop("ID", axis=1)

# Target
target = "SeriousDlqin2yrs"

# ------------------------------
# 1. HANDLE MISSING VALUES
# ------------------------------

# Fill MonthlyIncome missing with median
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

# Fill NumberOfDependents with mode
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
    df["NumberOfDependents"].mode()[0]
)

# ------------------------------
# 2. FEATURE SPLIT
# ------------------------------

X = df.drop(target, axis=1)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------
# 3. FEATURE SCALING
# ------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 4. TRAIN LOGISTIC REGRESSION
# ------------------------------

log_reg = LogisticRegression(max_iter=500, class_weight="balanced")  
# class_weight helps with imbalance

log_reg.fit(X_train_scaled, y_train)

# ------------------------------
# 5. EVALUATION
# ------------------------------

# Probabilities
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# ROC-AUC Score
auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", auc)

# Convert to binary predictions
y_pred = (y_pred_proba > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# 6. PLOT ROC CURVE
# ------------------------------

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.show()





