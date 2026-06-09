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
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")



df = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df.head()


print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())



plt.figure(figsize=(8,5))
sns.histplot(df['annual_income'], kde=True)
plt.title("Annual Income Distribution")
plt.show()



plt.figure(figsize=(8,5))
sns.histplot(df['credit_score'], kde=True)
plt.title("Credit Score Distribution")
plt.show()



plt.figure(figsize=(6,4))
sns.countplot(x='loan_paid_back', data=df)
plt.title("Loan Paid Back Count")
plt.show()



sns.countplot(data=df, x='gender')
plt.title("Gender Distribution")
plt.show()



sns.countplot(data=df, x='marital_status')
plt.title("Marital Status Distribution")
plt.show()



plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='annual_income', y='loan_amount', hue='loan_paid_back')
plt.title("Annual Income vs Loan Amount")
plt.show()



plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x='credit_score', y='interest_rate', hue='loan_paid_back')
plt.title("Credit Score vs Interest Rate")
plt.show()



plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='loan_paid_back', y='debt_to_income_ratio')
plt.title("DTI Ratio vs Loan Paid Back")
plt.show()



plt.figure(figsize=(7,5))
sns.countplot(data=df, x='education_level', hue='loan_paid_back')
plt.xticks(rotation=45)
plt.title("Education Level vs Loan Paid Back")
plt.show()



plt.figure(figsize=(10,5))
sns.countplot(data=df, x='loan_purpose', hue='loan_paid_back')
plt.xticks(rotation=45)
plt.title("Loan Purpose vs Default Rate")
plt.show()




numeric_df = df.select_dtypes(include=['int64', 'float64'])


plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap (Numerical Features Only)")
plt.show()



sns.pairplot(df[['annual_income','credit_score','loan_amount','debt_to_income_ratio','interest_rate','loan_paid_back']], hue='loan_paid_back')
plt.show()



df.shape


df.isnull().sum()


df.columns


df.dtypes


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# ğŸ”¹ Step 3: à¦ªà§�à¦°à¦¿à¦¨à§�à¦Ÿ à¦•à¦°à§‹
print("ğŸ”¢ Numerical Columns:")
print(numerical_cols.tolist())

print("\nğŸ”  Categorical Columns:")
print(categorical_cols.tolist())




