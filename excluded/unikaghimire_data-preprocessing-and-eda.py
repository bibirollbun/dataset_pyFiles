import pandas as pd


# Loading our dataset

df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")


# Shape 

print(f"The shape of our dataset : {df.shape}")


df.head(10)


import seaborn as sns
import matplotlib.pyplot as plt
print("\nLoan Paid Back Value Counts:")
print(df['loan_paid_back'].value_counts())

sns.countplot(data=df, x='loan_paid_back')
plt.title("Loan Paid Back Distribution")
plt.show()


balance = df['loan_paid_back'].value_counts(normalize=True) * 100
print("\nPercentage distribution of loan_paid_back:")
print(balance)


TARGET = 'loan_paid_back'
TRAIN_SAMPLE = 100000  
TEST_SIZE = 100000
RANDOM_STATE = 42

df_0 = df[df[TARGET] == 0]
df_1 = df[df[TARGET] == 1]

# Create balanced training dataset
df_0_train = df_0.sample(n=TRAIN_SAMPLE, random_state=RANDOM_STATE)
df_1_train = df_1.sample(n=TRAIN_SAMPLE, random_state=RANDOM_STATE)

df_balanced = pd.concat([df_0_train, df_1_train]) \
                  .sample(frac=1, random_state=RANDOM_STATE) \
                  .reset_index(drop=True)

# Remaining data for test set
df_remaining = df.drop(df_balanced.index)

# Sample 1 lakh rows for test set
df_test = df_remaining.sample(n=TEST_SIZE, random_state=RANDOM_STATE) \
                       .reset_index(drop=True)

print("Balanced dataset distribution:")
print(df_balanced[TARGET].value_counts())

print("\nTest dataset distribution:")
print(df_test[TARGET].value_counts())

print("\nSizes:")
print("Balanced:", df_balanced.shape)
print("Test:", df_test.shape)


# Overwrite df with the balanced dataset
df = df_balanced

# Optional: check
print(df['loan_paid_back'].value_counts())


df.shape


import matplotlib.pyplot as plt
import seaborn as sns

print("\nLoan Paid Back Value Counts:")
print(df['loan_paid_back'].value_counts())

sns.countplot(data=df, x='loan_paid_back')
plt.title("Loan Paid Back Distribution")
plt.show()


balance = df['loan_paid_back'].value_counts(normalize=True) * 100
print("\nPercentage distribution of loan_paid_back:")
print(balance)


# Let's look at our columns.

columns = df.columns
print("The columns are : ")
i = 0 
for cols in columns:
    print(f"\n {cols}")
    i = i + 1

print(f"\n In total, there are {i} columns.")


# Let's see if any of our columns have any null values

df.isnull().sum()


df = df.drop('id', axis=1)


df_lr = df.copy()
df_lr.shape


df.dtypes


# Let's work on 'object' attributes and look if we can apply
# one-hot encoding or label-encoding.

obj_columns = df.select_dtypes(object).columns

for col in obj_columns:
    print(df[col].value_counts())
    print("\n")


columns_and_cardinality = {}

def get_cardinality(columns):
    for col in columns:
        unique_values = len(df[col].unique())
        columns_and_cardinality[col] = unique_values

    return columns_and_cardinality


col_card = get_cardinality(obj_columns)
print(col_card)


# Step 1: Automatically detect categorical columns (object type)
obj_columns = df.select_dtypes(include=['object']).columns.tolist()

# Step 2: Remove columns you don't want to encode
skip_cols = ["grade_subgrade", "education_level"]
obj_columns = [col for col in obj_columns if col not in skip_cols]

print("Columns to be one-hot encoded:", obj_columns)

# Step 3: Apply one-hot encoding safely
df = pd.get_dummies(df, columns=obj_columns, drop_first=True)

# Step 4: Check the new dataframe
print(df.head())
print("Shape after encoding:", df.shape)



# Check if column exists
if 'grade_subgrade' in df.columns:
    # Extract grade and subgrade
    df['grade'] = df['grade_subgrade'].str[0]
    df['subgrade'] = df['grade_subgrade'].str[1:].astype(int)

    # Map grades to numbers
    grade_map = {'A': 1,'B': 2,'C': 3,'D': 4,'E': 5,'F': 6,'G': 7}
    df['grade'] = df['grade'].map(grade_map)

    # Create single encoded column
    df['grade_subgrade_encoded'] = df['grade'] * 10 + df['subgrade']

    # Drop original column
    df = df.drop('grade_subgrade', axis=1)
    print("Ordinal encoding applied successfully.")
else:
    print("Column 'grade_subgrade' does not exist. Maybe it was already dropped or encoded.")


df.grade.value_counts()


# # Ordinal Encoding on education_level

df['education_level'].value_counts()
education_level = {'Other': 0, 'High School': 1, "Bachelor's": 2, "Master's": 3, 'PhD': 4}
df['education_level'] = df['education_level'].map(education_level)
df.head(10)



# Let's look at our columns.

columns = df.columns
print("The columns are : ")
i = 0 
for cols in columns:
    print(f"\n {cols}")
    i = i + 1

print(f"\n In total, there are {i} columns.")


df_lr = df.copy()


import pandas as pd

# Copy the df_test to avoid overwriting
df_test_processed = df_test.copy()

# ===============================
# Step 1: One-Hot Encoding for categorical columns
# ===============================
# Columns that were one-hot encoded in training
one_hot_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose']
# Columns we excluded from one-hot encoding
skip_cols = ['grade_subgrade', 'education_level']

# Apply one-hot encoding
for col in one_hot_cols:
    if col in df_test_processed.columns:
        dummies = pd.get_dummies(df_test_processed[col], prefix=col, drop_first=True)
        df_test_processed = pd.concat([df_test_processed, dummies], axis=1)
        df_test_processed.drop(col, axis=1, inplace=True)

# ===============================
# Step 2: Ordinal Encoding
# ===============================
# Grade & Subgrade
if 'grade_subgrade' in df_test_processed.columns:
    df_test_processed['grade'] = df_test_processed['grade_subgrade'].str[0].map({
        'A': 1,'B': 2,'C': 3,'D': 4,'E': 5,'F': 6,'G': 7
    })
    df_test_processed['subgrade'] = df_test_processed['grade_subgrade'].str[1:].astype(int)
    df_test_processed['grade_subgrade_encoded'] = df_test_processed['grade'] * 10 + df_test_processed['subgrade']
    df_test_processed.drop('grade_subgrade', axis=1, inplace=True)

# Education Level
education_map = {'Other': 0, 'High School': 1, "Bachelor's": 2, "Master's": 3, 'PhD': 4}
if 'education_level' in df_test_processed.columns:
    df_test_processed['education_level'] = df_test_processed['education_level'].map(education_map)

# ===============================
# Step 3: Engineered Features
# ===============================
df_test_processed['income_loan_ratio'] = df_test_processed['annual_income'] / (df_test_processed['loan_amount'] + 1e-6)
df_test_processed['debt_load'] = df_test_processed['debt_to_income_ratio'] * df_test_processed['loan_amount']
df_test_processed['risk_score'] = df_test_processed['debt_to_income_ratio'] / (df_test_processed['credit_score'] + 1e-6)

# ===============================
# Step 4: Align columns with training df
# ===============================
# Ensure same columns exist (especially after one-hot encoding)
missing_cols = set(df.columns) - set(df_test_processed.columns)
for col in missing_cols:
    df_test_processed[col] = 0

# Reorder columns to match training df
df_test_processed = df_test_processed[df.columns]

print("Test data processed. Shape:", df_test_processed.shape)
df_test_processed.head()


num_columns = df.select_dtypes(int).columns
for cols in num_columns:
    print(cols)

num_columns


df[num_columns].describe()


import matplotlib.pyplot as plt

# Number of columns to show per row
cols_per_row = 3
num_rows = (len(num_columns) + cols_per_row - 1) // cols_per_row  # ceiling division

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(3*cols_per_row, 4*num_rows))
axes = axes.flatten()  # flatten in case of multiple rows

for i, col in enumerate(num_columns):
    axes[i].hist(df[col], bins=30, color='skyblue', edgecolor='black')
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

# Turn off any extra empty subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Number of boxplots per row
cols_per_row = 3
num_rows = (len(num_columns) + cols_per_row - 1) // cols_per_row  # ceiling division

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(3*cols_per_row, 4*num_rows))
axes = axes.flatten()  # flatten axes for easy iteration

for i, col in enumerate(num_columns):
    sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
    axes[i].set_title(f'{col} Boxplot')

# Turn off any extra empty subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# select object columns
obj_cols = df.select_dtypes(object).columns

# loop through them
for col in obj_cols:
    print(df[col].value_counts())
    print('\n')


import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Numerical vs Target (grid layout)
# -------------------------------
cols_per_row = 3
num_rows = (len(num_columns) + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(5*cols_per_row, 4*num_rows))
axes = axes.flatten()

for i, col in enumerate(num_columns):
    sns.boxplot(x='loan_paid_back', y=col, data=df, ax=axes[i], palette='Set2', hue=None)
    axes[i].set_title(f'{col} vs Loan Paid Back')

# turn off unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

    # Boxplot
    sns.boxplot(x='loan_paid_back', y=col, data=df, ax=axes[0], palette='Set2')
    axes[0].set_title(f'{col} vs Loan Paid Back (Boxplot)')

    # Violinplot
    sns.violinplot(x='loan_paid_back', y=col, data=df, ax=axes[1], palette='Set3')
    axes[1].set_title(f'{col} vs Loan Paid Back (Violinplot)')

    plt.tight_layout()
    plt.show()


import math
cat_cols = [
    'gender_Male',
    'gender_Other',
    'marital_status_Married',
    'marital_status_Single',
    'marital_status_Widowed',
    'employment_status_Retired',
    'employment_status_Self-employed',
    'employment_status_Student',
    'employment_status_Unemployed',
    'loan_purpose_Car',
    'loan_purpose_Debt consolidation',
    'loan_purpose_Education',
    'loan_purpose_Home',
    'loan_purpose_Medical',
    'loan_purpose_Other',
    'loan_purpose_Vacation'
]

# Ensure columns exist
cat_cols = [col for col in cat_cols if col in df.columns]

cols_per_row = 3
num_rows = math.ceil(len(cat_cols) / cols_per_row)

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(6*cols_per_row, 4*num_rows))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    sns.countplot(x=col, hue='loan_paid_back', data=df, palette='Set2', ax=axes[i])
    axes[i].set_title(f'{col} vs Loan Paid Back')
    axes[i].tick_params(axis='x', rotation=45)

# Turn off any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between Numerical Features")
plt.show()


# EDA: 

"""

- Preprocessing: One hot, label encoding, 
- Univariate data analysis: Analysing the distribution 
- Bivariate data analysis



- Correlation heatmap
- PCA
- LDA

- Feature Engineering

### Modeling

5 Classification algorithm:

    - Decision Tree (Tree Based Model)
    - Random Forest (Bagging Based Algorithm)
    - XGBoost (Boosting)
    - LGBM Classifier
    - CatBoost
    - Linear Regression

- Training vs Validation line graph to check overfitting
- Cross Validation: K-fold or Stratified K-folding
- Regularization techniques: L1, L2
- Optuna: Hyperparameter tuning 
"""


from scipy.stats import skew

numeric_cols = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate']
skewed = df[numeric_cols].apply(lambda x: skew(x))
print(skewed)


from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import math

num_cols = [col for col in numeric_cols if col in df.columns]

cols_per_row = 2
num_rows = math.ceil(len(num_cols) / cols_per_row)

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(6*cols_per_row, 4*num_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, bins=30, color='skyblue', edgecolor='black', ax=axes[i])
    axes[i].set_title(f"{col} (Skew: {skew(df[col]):.2f})")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

# Turn off any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



df.shape


import pandas as pd
import numpy as np

# Columns to handle outliers
numeric_cols = ['annual_income','debt_to_income_ratio','loan_amount','interest_rate']

# Cap outliers at 1st and 99th percentile
for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# Check the new min and max
df[numeric_cols].describe()


from scipy.stats import skew

numeric_cols = ['annual_income','debt_to_income_ratio','credit_score','loan_amount','interest_rate']
skewed = df[numeric_cols].apply(lambda x: skew(x))
print(skewed)


import matplotlib.pyplot as plt
import seaborn as sns

# Number of boxplots per row
cols_per_row = 3
num_rows = (len(num_columns) + cols_per_row - 1) // cols_per_row  # ceiling division

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(3*cols_per_row, 4*num_rows))
axes = axes.flatten()  # flatten axes for easy iteration

for i, col in enumerate(num_columns):
    sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
    axes[i].set_title(f'{col} Boxplot')

# Turn off any extra empty subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Numerical vs Target (grid layout)
# -------------------------------
cols_per_row = 3
num_rows = (len(num_columns) + cols_per_row - 1) // cols_per_row

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(5*cols_per_row, 4*num_rows))
axes = axes.flatten()

for i, col in enumerate(num_columns):
    sns.boxplot(x='loan_paid_back', y=col, data=df, ax=axes[i], palette='Set2', hue=None)
    axes[i].set_title(f'{col} vs Loan Paid Back')

# turn off unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()



num_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']

for col in num_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

    # Boxplot
    sns.boxplot(x='loan_paid_back', y=col, data=df, ax=axes[0], palette='Set2')
    axes[0].set_title(f'{col} vs Loan Paid Back (Boxplot)')

    # Violinplot
    sns.violinplot(x='loan_paid_back', y=col, data=df, ax=axes[1], palette='Set3')
    axes[1].set_title(f'{col} vs Loan Paid Back (Violinplot)')

    plt.tight_layout()
    plt.show()
    


num_cols = [col for col in numeric_cols if col in df.columns]

cols_per_row = 2
num_rows = math.ceil(len(num_cols) / cols_per_row)

fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(6*cols_per_row, 4*num_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, bins=30, color='skyblue', edgecolor='black', ax=axes[i])
    axes[i].set_title(f"{col} (Skew: {skew(df[col]):.2f})")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

# Turn off any unused subplots
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


# Income to Loan Ratio
# To see how big the loan is to income
# Higher ratio means more risk of not paying back

df['income_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1e-6) 

# Debt Load
# Combines debt-to-income ratio and loan amount.
# Shows total financial burden of the borrower.

df['debt_load'] = df['debt_to_income_ratio'] * df['loan_amount']

# Risk Score 
# (Debt-to-Income / Credit Score)
# Combines debt-to-income ratio and credit score.
# Higher score → higher risk of default.

df['risk_score'] = df['debt_to_income_ratio'] / (df['credit_score'] + 1e-6)


# def simplify_employment(row):
#     if row['employment_status_Retired'] == 1 or 
#     row['employment_status_Unemployed'] == 1:
#         return 'Other'
#     elif row['employment_status_Self-employed'] == 1 or 
#     row['employment_status_Student'] == 1:
#         return 'Employed/Student'
#     else:
#         return 'Employed'
    
# df['employment_simplified'] = df.apply(simplify_employment, axis=1)

# # Group Loan Purposes
# def group_loan_purpose(row):
#     if row['loan_purpose_Car'] == 1 or row['loan_purpose_Vacation'] == 1 
#     or row['loan_purpose_Medical'] == 1:
#         return 'Personal'
#     elif row['loan_purpose_Education'] == 1:
#         return 'Education'
#     elif row['loan_purpose_Home'] == 1:
#         return 'Home'
#     else:
#         return 'Other'
    
# df['loan_purpose_grouped'] = df.apply(group_loan_purpose, axis=1)

# # Interaction Features
# df['edu_income_interaction'] = df['annual_income'] * df['education_level'].astype(int) 
# df['purpose_loan_interaction'] = df['loan_amount'] * df['loan_purpose_grouped'].map({
#     'Personal': 1, 'Education': 2, 'Home': 3, 'Other': 0
# })


