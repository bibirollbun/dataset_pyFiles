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



#Let's show the 10 row of our data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")




#head of our dataset
train_df.head(5)
test_df.head(5)



#dimension of dataset
train_df.shape
test_df.shape #Only one column is removed




#description
train_df.describe().T #Transpose



#dataset information
train_df.info()
test_df.info()



#distibution of data
import matplotlib.pyplot as plt

numeric_cols = train_df.select_dtypes(include=['int64']).columns
rows = 3
cols = 6
fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
axes = axes.flatten()  

for i, col in enumerate(numeric_cols):
    train_df[col].hist(ax=axes[i], bins=20)
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')


for j in range(i + 1, rows * cols):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
    



import seaborn as sns
#Corrolation
num_df = train_df.select_dtypes(include='number')
# Pearson
plt.figure(figsize=(10,8))
sns.heatmap(num_df.corr(method='pearson'), annot=True, cmap='coolwarm')
plt.title("Heatmap (Pearson Correlation)")
plt.show()

# Spearman
plt.figure(figsize=(10,8))
sns.heatmap(num_df.corr(method='spearman'), annot=True, cmap='coolwarm')
plt.title("Heatmap (Spearman Correlation)")
plt.show()

pearson_corr = num_df.corr(method='pearson')['y'].drop('y')
spearman_corr = num_df.corr(method='spearman')['y'].drop('y')




import matplotlib.pyplot as plt

y_counts = train_df['y'].value_counts()
y_percent = train_df['y'].value_counts(normalize=True) * 100

print("Class distribution:")
print(y_counts)
print("\nPercentage:")
print(y_percent.round(2))

#Charts
plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='y')
plt.title('Target Variable Distribution (y)')
plt.xlabel('Subscribed to Term Deposit')
plt.ylabel('Count')
plt.xticks([0, 1], ['No', 'Yes'])
plt.grid(axis='y')
plt.show()




import math
categorical_cols = train_df.select_dtypes(include=['object']).columns.drop('y', errors='ignore')

#subplot
n_cols = 2  
n_rows = math.ceil(len(categorical_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*4))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    cat_order = train_df[col].value_counts().index
    sns.countplot(data=train_df, x=col, hue='y', order=cat_order, ax=axes[i])
    axes[i].set_title(f'{col} vs y')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='y', labels=['No', 'Yes'])
    axes[i].grid(axis='y', alpha=0.3)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()




num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(data=train_df, x='y', y=col, ax=axes[i])
    axes[i].set_title(f'{col} vs y')
    axes[i].set_xlabel('y (Subscribed)')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()



#outlier detection
outlier_summary = {}

for col in num_cols:
    Q1 = train_df[col].quantile(0.25)
    Q3 = train_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = train_df[(train_df[col] < lower_bound) | (train_df[col] > upper_bound)][col]
    outlier_summary[col] = {
        'num_outliers': outliers.count(),
        'percent_outliers': 100 * outliers.count() / len(train_df),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

outlier_df = pd.DataFrame(outlier_summary).T.sort_values(by='percent_outliers', ascending=False)
outlier_df



missing = train_df.isnull().sum()
missing_percent = (missing / len(train_df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing Percent': missing_percent.round(2)
})

missing_df = missing_df[missing_df['Missing Count'] > 0]
missing_df
cat_cols = train_df.select_dtypes(include=['object']).columns

unknown_summary = {}

for col in cat_cols:
    count = (train_df[col] == 'unknown').sum()
    if count > 0:
        unknown_summary[col] = {
            'Unknown Count': count,
            'Unknown Percent': round(100 * count / len(train_df), 2)
        }

unknown_df = pd.DataFrame(unknown_summary).T.sort_values(by='Unknown Percent', ascending=False)
unknown_df


