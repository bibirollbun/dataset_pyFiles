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

train = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction-dataset/train.csv")
test = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction-dataset/test.csv")

print(train.shape)  # (200000, 202) -> 200k rows, 200 features + ID_code + target
print(test.shape)   # (200000, 201)



print(train.columns)       # See column names
print(train.dtypes.value_counts())  # Check data types
print(train.head())        # View first few rows



print(train['target'].value_counts())
print(train['target'].value_counts(normalize=True))  # For imbalance



print(train.isnull().sum().sum())  # Should be 0 — no missing values in this dataset
print(test.isnull().sum().sum())



nunique = train.drop(['ID_code', 'target'], axis=1).nunique()
constant_cols = nunique[nunique == 1].index.tolist()
print("Constant columns:", constant_cols)



train.drop(columns=constant_cols, inplace=True)
test.drop(columns=constant_cols, inplace=True)



def get_duplicate_columns(df):
    duplicates = set()
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicates.add(cols[j])
    return list(duplicates)

dup_cols = get_duplicate_columns(train.drop(['ID_code', 'target'], axis=1))
print("Duplicated columns:", dup_cols)

train.drop(columns=dup_cols, inplace=True)
test.drop(columns=dup_cols, inplace=True)



import matplotlib.pyplot as plt
import seaborn as sns

sample_var = 'var_0'

sns.histplot(train[sample_var], bins=100, kde=True)
plt.title(f'Distribution of {sample_var}')
plt.show()



def get_duplicate_columns(df):
    duplicates = set()
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicates.add(cols[j])
    return list(duplicates)

dup_cols = get_duplicate_columns(train.drop(['ID_code', 'target'], axis=1))
print("Duplicated columns:", dup_cols)

# Optionally drop
train.drop(columns=dup_cols, inplace=True)
test.drop(columns=dup_cols, inplace=True)



import pandas as pd
import numpy as np

# Assuming your dataframe is 'train'

# 1. Check for NaNs or missing values
print("Missing values per column:")
print(train.isna().sum()[train.isna().sum() > 0])

# 2. Check for infinite values
print("\nInfinite values per column:")
print(np.isinf(train.select_dtypes(include=[np.number])).sum())

# 3. Check for duplicates by rows (full duplicates)
num_duplicates = train.duplicated().sum()
print(f"\nNumber of fully duplicated rows: {num_duplicates}")

# 4. Check for constant columns (already done, but for completeness)
constant_cols = [col for col in train.columns if train[col].nunique() == 1]
print(f"\nConstant columns: {constant_cols}")

# 5. Check for outliers or suspiciously extreme values by feature
# Define a threshold for outliers, e.g., values beyond 5 standard deviations from mean
outlier_summary = {}
for col in train.select_dtypes(include=[np.number]).columns:
    mean = train[col].mean()
    std = train[col].std()
    outliers = train[(train[col] > mean + 5*std) | (train[col] < mean - 5*std)]
    outlier_summary[col] = len(outliers)

print("\nOutlier counts (values beyond 5 std deviations):")
print({k: v for k, v in outlier_summary.items() if v > 0})

# 6. Check for impossible or suspicious values depending on context
# Example: Negative values in features that should be positive (if any)
# (Assuming vars should be positive, modify if necessary)
neg_counts = (train.select_dtypes(include=[np.number]) < 0).sum()
print("\nCount of negative values per feature:")
print(neg_counts[neg_counts > 0])



import matplotlib.pyplot as plt
import seaborn as sns

# Plot histogram for a sample of variables with many negative values
sample_vars = ['var_1', 'var_5', 'var_8', 'var_193']

for var in sample_vars:
    plt.figure(figsize=(8, 4))
    sns.histplot(train[var], bins=50, kde=True)
    plt.title(f'Distribution of {var}')
    plt.show()



# Step 7.1: Basic stats for each feature
feature_cols = [col for col in train.columns if col not in ['ID_code', 'target']]
stats = train[feature_cols].describe().T
print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

# Step 7.2: Correlation matrix (Pearson)
corr_matrix = train[feature_cols].corr()
print(corr_matrix)

# Optional: Plot correlation heatmap for the first 30 variables to avoid huge plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix.iloc[:30, :30], cmap='coolwarm', center=0, annot=False)
plt.title('Correlation heatmap (first 30 features)')
plt.show()

# Step 7.3: Correlation with target variable
target_corr = train[feature_cols].corrwith(train['target']).sort_values(ascending=False)
print("Top 10 features correlated with target:")
print(target_corr.head(10))
print("Bottom 10 features correlated with target:")
print(target_corr.tail(10))



from scipy.stats import skew

skewness = train[feature_cols].skew()
skewed_feats = skewness[abs(skewness) > 1].index  # Threshold can be adjusted

print(f"Number of highly skewed features: {len(skewed_feats)}")
print(skewed_feats)

# Example: Apply log1p to skewed features with all positive values
import numpy as np

for feat in skewed_feats:
    if (train[feat] > 0).all():
        train[feat] = np.log1p(train[feat])
        test[feat] = np.log1p(test[feat])



target_corr = train[feature_cols].corrwith(train['target']).sort_values(ascending=False)
print("Top 10 features correlated with target:")
print(target_corr.head(10))
print("\nBottom 10 features correlated with target:")
print(target_corr.tail(10))



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[feature_cols])
test_scaled = scaler.transform(test[feature_cols])


