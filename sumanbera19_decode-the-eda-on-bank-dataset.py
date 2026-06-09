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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# -------------------------------
# Step 1: Load and preprocess data
# -------------------------------

# Example: Load your data
# ğŸ“¥ Load data
df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")
df


print("ğŸ”� Dataset Info:\n")
print(df.info())


print("\nğŸ“� Shape of Data:", df.shape)
print("\nğŸ“Š First 5 Rows:\n", df.head())


print("\nğŸ“ˆ Numerical Columns Summary:\n")
df.describe()



print("\nğŸ§© Missing Values:\n")
df.isnull().sum()



cat_cols = df.select_dtypes(include='object').columns
print("\nğŸ”¢ Unique Values per Categorical Column:")
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique values")



print("\nğŸ�¯ Target Variable 'y' Distribution:\n", df['y'].value_counts())
sns.countplot(data=df, x='y')
plt.title("Target Variable Distribution (y)")
plt.show()


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()


num_cols = df.select_dtypes(include='number').columns.drop(['id', 'y'])
df[num_cols].hist(figsize=(15, 10), bins=30)
plt.suptitle("Numerical Feature Distributions")
plt.show()



for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot of {col}")
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, hue='y')
    plt.title(f"{col} vs Target (y)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, 65, 100],
                         labels=['18-25','26-35','36-45','46-55','56-65','65+'])

age_target = df.groupby('age_group')['y'].value_counts(normalize=True).unstack()
age_target.plot(kind='bar', stacked=True, figsize=(10,6), colormap='Set2')
plt.title("Age Group vs Target (y)")
plt.ylabel("Proportion")
plt.show()



job_target = df.groupby('job')['y'].value_counts(normalize=True).unstack()
job_target.plot(kind='bar', stacked=True, figsize=(12,6), colormap='coolwarm')
plt.title("Job vs Target (y)")
plt.xticks(rotation=45)
plt.ylabel("Proportion")
plt.show()



sns.histplot(data=df, x='balance', bins=50, hue='y', element='step', stat='density')
plt.title("Balance Distribution by Target")
plt.xlim(-2000, 40000)
plt.show()



# Contact Type
sns.countplot(data=df, x='contact', hue='y')
plt.title("Contact Method vs Target")
plt.show()

# Month
sns.countplot(data=df, x='month', hue='y',
              order=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
plt.title("Month vs Target (y)")
plt.show()




# âœ… Target Pie Chart
y_counts = df['y'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(y_counts, labels=y_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral'])
plt.title("Target Variable (y) Distribution")
plt.axis('equal')
plt.show()

# âœ… Job-wise Pie Chart
job_group = df.groupby('job')['y'].value_counts(normalize=True).unstack()
print("âœ… Job-wise Pie Chart")
for job in job_group.index:
    plt.figure(figsize=(4, 4))
    plt.pie(job_group.loc[job], labels=job_group.columns, autopct='%1.1f%%', startangle=90)
    plt.title(f"'y' Distribution for Job: {job}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# âœ… Marital Status-wise Pie Chart
marital_group = df.groupby('marital')['y'].value_counts(normalize=True).unstack()
print("âœ… Marital Status-wise Pie Chart")
for status in marital_group.index:
    plt.figure(figsize=(4, 4))
    plt.pie(marital_group.loc[status], labels=marital_group.columns, autopct='%1.1f%%', startangle=90)
    plt.title(f"'y' Distribution for Marital Status: {status}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()



# âœ… Housing-wise Pie
housing_pie = df.groupby('housing')['y'].value_counts(normalize=True).unstack()
print("âœ… Housing-wise Pie")

for hs in housing_pie.index:
    plt.figure(figsize=(4, 4))
    plt.pie(housing_pie.loc[hs], labels=housing_pie.columns, autopct='%1.1f%%')
    plt.title(f"'y' Distribution for Housing: {hs}")
    plt.axis('equal')
    plt.show()

# âœ… Loan-wise Pie
loan_pie = df.groupby('loan')['y'].value_counts(normalize=True).unstack()
print("âœ… Loan-wise Pie")
for ln in loan_pie.index:
    plt.figure(figsize=(4, 4))
    plt.pie(loan_pie.loc[ln], labels=loan_pie.columns, autopct='%1.1f%%')
    plt.title(f"'y' Distribution for Loan: {ln}")
    plt.axis('equal')
    plt.show()

# âœ… Contact Method Countplot
sns.countplot(data=df, x='contact', hue='y')
plt.title("Contact Method vs Target")
plt.show()

# âœ… Previous Campaign Outcome Pie
poutcome_pie = df.groupby('poutcome')['y'].value_counts(normalize=True).unstack()
print("âœ… Previous Campaign Outcome Pie")
for po in poutcome_pie.index:
    plt.figure(figsize=(4, 4))
    plt.pie(poutcome_pie.loc[po], labels=poutcome_pie.columns, autopct='%1.1f%%')
    plt.title(f"'y' Distribution for Previous Outcome: {po}")
    plt.axis('equal')
    plt.show()

# âœ… Month-wise Pie
month_group = df.groupby('month')['y'].value_counts(normalize=True).unstack()
print(" Month-wise Pie")
for m in month_group.index:
    plt.figure(figsize=(4, 4))
    plt.pie(month_group.loc[m], labels=month_group.columns, autopct='%1.1f%%')
    plt.title(f"'y' Distribution for Month: {m}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()



# Downsample (optional but helps performance)
df_sample = df.sample(n=10000, random_state=42)

# âœ… Make sure 'y' is categorical (not numeric)
df_sample['y'] = df_sample['y'].astype(str)

# âœ… Select only numeric columns
numeric_cols = df_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()

# âœ… Include target 'y' separately
pair_df = df_sample[numeric_cols + ['y']]

# âœ… Plot pairplot
sns.pairplot(pair_df, hue='y', diag_kind='hist', palette='coolwarm')
plt.suptitle("ğŸ“Š Pair Plot of Numerical Features (Colored by y)", y=1.02)
plt.show()


