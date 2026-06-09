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


train_path = '/kaggle/input/playground-series-s5e12/train.csv'
test_path = '/kaggle/input/playground-series-s5e12/test.csv'
submission_path = '/kaggle/input/playground-series-s5e12/sample_submission.csv'


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
submission = pd.read_csv(submission_path)


print("--- Train Head ---")
print(train.head())
print("\n--- Train Info ---")
print(train.info())
print("\n--- Train Shape ---")
print(train.shape)
print("\n--- Test Head ---")
print(test.head())
print("\n--- Test Shape ---")
print(test.shape)
print("\n--- Submission Head ---")
print(submission.head())



# -------------------------
# Target distribution
# -------------------------
print("=== TARGET DISTRIBUTION ===")
print(train['diagnosed_diabetes'].value_counts(dropna=False))
print(train['diagnosed_diabetes'].value_counts(normalize=True)*100)
print("\n")

# -------------------------
# Missing values (Train & Test)
# -------------------------
print("=== MISSING VALUES (TRAIN) ===")
missing_train = train.isnull().sum().sort_values(ascending=False)
print(missing_train[missing_train > 0])

print("\n=== MISSING VALUES (TEST) ===")
missing_test = test.isnull().sum().sort_values(ascending=False)
print(missing_test[missing_test > 0])
print("\n")

# -------------------------
# Describe numeric columns
# -------------------------
print("=== NUMERIC SUMMARY ===")
print(train.describe().T)
print("\n")

# -------------------------
# Unique values in categorical columns
# -------------------------
cat_cols = train.select_dtypes(include=['object']).columns
print("=== CATEGORICAL UNIQUE COUNTS ===")
for col in cat_cols:
    print(f"\nColumn: {col}")
    print("Unique:", train[col].nunique())
    print(train[col].value_counts().head(3))
print("\n")

# -------------------------
# Sanity checks for invalid ranges
# -------------------------
print("=== RANGE CHECKS ===")
def check_range(col, low, high):
    bad = train[(train[col] < low) | (train[col] > high)]
    print(f"{col}: {bad.shape[0]} invalid values")

check_range('age', 0, 120)
check_range('bmi', 10, 70)
check_range('systolic_bp', 50, 300)
check_range('diastolic_bp', 30, 200)
check_range('waist_to_hip_ratio', 0.4, 1.5)
check_range('sleep_hours_per_day', 0, 24)
check_range('physical_activity_minutes_per_week', 0, 10000)
check_range('alcohol_consumption_per_week', 0, 200)
print("\n")

# -------------------------
# Duplicate ID checks
# -------------------------
print("=== DUPLICATE ID CHECK ===")
print("Duplicates in train:", train['id'].duplicated().sum())
print("Duplicates in test :", test['id'].duplicated().sum())
print("Overlap train vs test IDs:",
      len(set(train['id']).intersection(set(test['id']))))
print("\n")

# -------------------------
# Constant or near-constant columns
# -------------------------
print("=== CONSTANT / NEAR-CONSTANT COLUMNS ===")
for col in train.columns:
    freq = train[col].value_counts(normalize=True, dropna=False)
    if freq.iloc[0] > 0.99:
        print(f"{col} â†’ {freq.iloc[0]*100:.2f}% same value")
print("\n")

# -------------------------
# Correlation with target
# -------------------------
print("=== CORRELATION WITH TARGET ===")
num_cols = train.select_dtypes(include=[np.number]).columns.drop("diagnosed_diabetes")

corrs = train[num_cols].corrwith(train['diagnosed_diabetes']).abs().sort_values(ascending=False)
print(corrs.head(10))
print("\n")




import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
if "id" in num_cols:
    num_cols.remove("id")
if "diagnosed_diabetes" in num_cols:
    num_cols.remove("diagnosed_diabetes")   
cat_cols = train.select_dtypes(include=["object"]).columns.tolist()

print("\nğŸ“Œ NUMERICAL COLUMNS:", num_cols)

num_plots = len(num_cols)
plots_per_row = 5
num_rows = (num_plots + plots_per_row - 1) // plots_per_row 

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(20, 4 * num_rows))

if num_rows > 1:
    axes = axes.flatten()
elif num_rows == 1 and plots_per_row == 1:
    axes = [axes] # Ensure axes is iterable even for 1 plot

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.histplot(train[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout() 
plt.show()





def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)][col]
    print(f"{col}: {len(outliers)} outliers")
    return outliers

print("OUTLIER COUNTS")
for col in num_cols:
    detect_outliers(train, col)

num_plots = len(num_cols)
plots_per_row = 5
num_rows = (num_plots + plots_per_row - 1) // plots_per_row 

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(22, 3.5 * num_rows))

if num_rows > 1 and plots_per_row > 1:
    axes = axes.flatten()
elif num_rows == 1 and plots_per_row == 1:
    axes = [axes]
elif num_rows == 1 or plots_per_row == 1:
    axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(x=train[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    ax.set_xlabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout() 
plt.show()



print("\nğŸ“Œ CATEGORICAL COLUMNS:", cat_cols)

num_plots = len(cat_cols)
plots_per_row = 3
num_rows = (num_plots + plots_per_row - 1) // plots_per_row 

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(18, 5 * num_rows))

if num_rows > 1 and plots_per_row > 1:
    axes = axes.flatten()
elif num_rows == 1 and plots_per_row == 1:
    axes = [axes]
elif num_rows == 1 or plots_per_row == 1:
    axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    sns.countplot(x=train[col], ax=ax)
    ax.set_title(f"Countplot of {col}")
    ax.set_xlabel(col)
    ax.tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


num_plots = len(cat_cols)
plots_per_row = 3
num_rows = (num_plots + plots_per_row - 1) // plots_per_row 

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(18, 5 * num_rows))

if num_rows > 1 and plots_per_row > 1:
    axes = axes.flatten()
elif num_rows == 1 and plots_per_row == 1:
    axes = [axes]
elif num_rows == 1 or plots_per_row == 1:
    axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ax = axes[i]
    diabetes_rate = train.groupby(col)["diagnosed_diabetes"].mean()
    diabetes_rate.plot(kind="bar", ax=ax)
    ax.set_ylabel("Diabetes Rate")
    ax.set_title(f"Diabetes Rate by {col}")
    ax.tick_params(axis='x', rotation=45)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


num_plots = len(num_cols)
plots_per_row = 5
num_rows = (num_plots + plots_per_row - 1) 

fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(25, 5 * num_rows))

if num_rows > 1 and plots_per_row > 1:
    axes = axes.flatten()
elif num_rows == 1 and plots_per_row == 1:
    axes = [axes]
elif num_rows == 1 or plots_per_row == 1:
    axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    sns.boxplot(data=train, x="diagnosed_diabetes", y=col, ax=ax)
    ax.set_title(f"{col} vs Diabetes")
    ax.set_xlabel("Diagnosed Diabetes (0/1)")
    ax.set_ylabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,10))
sns.heatmap(train[num_cols + ["diagnosed_diabetes"]].corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()


corr_matrix = train[num_cols + ["diagnosed_diabetes"]].corr()

# Extract correlations with 'diagnosed_diabetes'
diabetes_corr = corr_matrix["diagnosed_diabetes"].drop("diagnosed_diabetes")


top_15_corr = diabetes_corr.abs().sort_values(ascending=False).head(15)

print(" Top 15 Features Correlated with 'diagnosed_diabetes' (by absolute value):")
print(top_15_corr)


cols_check = [
    "physical_activity_minutes_per_week",
    "sleep_hours_per_day",
    "screen_time_hours_per_day",
    "triglycerides",
    "diet_score"
]

train[cols_check].describe()



df = train.copy()

#  Physical activity - cap at 500 mins/week
df['physical_activity_minutes_per_week'] = df['physical_activity_minutes_per_week'].clip(upper=500)

# Screen time - cap at 14 hours/day
df['screen_time_hours_per_day'] = df['screen_time_hours_per_day'].clip(upper=14)

# apply log1p transform later to reduce skew
df['triglycerides_log'] = np.log1p(df['triglycerides'])


from sklearn.preprocessing import LabelEncoder

# Copy to avoid modifying original
train_enc = train.copy()
test_enc = test.copy()

# List of categorical columns
cat_cols = [
    'gender',
    'ethnicity',
    'education_level',
    'income_level',
    'smoking_status',
    'employment_status'
]


for col in cat_cols:
    le = LabelEncoder()

    # Fit on train + test combined (to avoid unseen labels)
    combined = pd.concat([train_enc[col], test_enc[col]], axis=0)

    le.fit(combined)

    train_enc[col] = le.transform(train_enc[col])
    test_enc[col] = le.transform(test_enc[col])

print("Encoding Completed! Shape:", train_enc.shape, test_enc.shape)



from sklearn.model_selection import train_test_split

# Drop ID & Target
X = train.drop(['diagnosed_diabetes', 'id'], axis=1)
y = train['diagnosed_diabetes']

# Stratified Split (very important because target is imbalanced)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    stratify=y, 
    random_state=42
)

print("Shapes:")
print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
print("y_train:", y_train.shape)
print("y_valid:", y_valid.shape)



import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ---------------------------
# 1ï¸�âƒ£ Prepare Data
# ---------------------------
X = train_enc.drop(['diagnosed_diabetes','id'], axis=1)
y = train_enc['diagnosed_diabetes']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 2ï¸�âƒ£ Train XGBoost
# ---------------------------
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

# ---------------------------
# 3ï¸�âƒ£ Predict and Evaluate
# ---------------------------
y_pred = xgb.predict(X_valid)
y_proba = xgb.predict_proba(X_valid)[:, 1]

acc = accuracy_score(y_valid, y_pred)
roc_auc = roc_auc_score(y_valid, y_proba)

print(f"XGBoost Validation Accuracy: {acc:.4f}")
print(f"XGBoost Validation ROC-AUC: {roc_auc:.4f}")

# ---------------------------
# 4ï¸�âƒ£ ROC Curve Plot
# ---------------------------
fpr, tpr, thresholds = roc_curve(y_valid, y_proba)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc_val:.4f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()




from sklearn.metrics import accuracy_score, roc_auc_score

# Full data
X_full = train_enc.drop(['diagnosed_diabetes','id'], axis=1)
y_full = train_enc['diagnosed_diabetes']

# Train XGBoost on full data
xgb_full = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_full.fit(X_full, y_full)

# Predict on full data
y_pred_full = xgb_full.predict(X_full)
y_proba_full = xgb_full.predict_proba(X_full)[:, 1]

# Evaluate
acc_full = accuracy_score(y_full, y_pred_full)
roc_auc_full = roc_auc_score(y_full, y_proba_full)

print(f"XGBoost Accuracy on full data: {acc_full:.4f}")
print(f"XGBoost ROC-AUC on full data: {roc_auc_full:.4f}")



X_test = test_enc.drop(['id'], axis=1)

# Predict on test data
test_preds = xgb_full.predict(X_test)

# Prepare submission
submission = submission.copy()
submission['diagnosed_diabetes'] = test_preds


submission.to_csv('xgb_submission.csv', index=False)
print("Submission saved as xgb_submission.csv")




