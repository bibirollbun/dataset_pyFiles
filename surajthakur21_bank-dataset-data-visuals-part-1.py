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


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.info()


test.info()


train.head(5)


train.drop(columns=['id'],inplace= True)


print("Train data shape:", train.shape)
print("Test data shape:", test.shape)


train.describe()


missing_values_train = train.isnull().sum()
print("\nMissing values in train data:")
print(missing_values_train)

missing_values_test = test.isnull().sum()
print("\nMissing values in test data:")
print(missing_values_test)


target_balance = train['y'].value_counts(normalize=True)
print("\nTarget variable balance (proportion):")
print(target_balance)


duplicate_rows_train = train.duplicated().sum()
print(f"\nNumber of duplicate rows in train data: {duplicate_rows_train}")

duplicate_rows_test = test.duplicated().sum()
print(f"\nNumber of duplicate rows in test data: {duplicate_rows_test}")


numeric_cols = train.select_dtypes(include=np.number).columns.tolist()

outlier_analysis = {}
for col in numeric_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outliers = train[(train[col] < lower_fence) | (train[col] > upper_fence)]
    outlier_count = outliers.shape[0]
    outlier_percentage = (outlier_count / len(train)) * 100

    outlier_analysis[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower Fence': lower_fence,
        'Upper Fence': upper_fence,
        'Outlier Count': outlier_count,
        'Outlier Percentage (%)': outlier_percentage
    }

outlier_analysis_df = pd.DataFrame(outlier_analysis).T
print("\nOutlier Analysis (IQR method):")
display(outlier_analysis_df)


unique_counts = train.nunique()
print("\nUnique value counts for each column:")
display(unique_counts)


num_features = train.select_dtypes(include= ['int64'])
num_features


cat_features = train.select_dtypes(include=['object'])
cat_features


import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("darkgrid")
sns.set_palette('husl')


import warnings
warnings.filterwarnings("ignore")


plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols):
    plt.subplot(3, 3, i + 1)
    sns.kdeplot(train[col],fill=True)
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(4, 2, figsize=(20, 15))

fig.suptitle("Distribution of each feature ", fontsize=20)

for i, num in enumerate(num_features.columns):
    row = i // 2
    col = i % 2
    sns.distplot(x=train[num], ax=ax[row, col])
    ax[row, col].set_title(f'Distribution of {num}')
    ax[row, col].set_xlabel(num)
    ax[row, col].set_ylabel('Index')

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()


fig, ax = plt.subplots(4, 2, figsize=(20, 15))

fig.suptitle("Distribution of each feature ", fontsize=20)

for i, num in enumerate(num_features.columns):
    row = i // 2
    col = i % 2
    sns.boxplot(x=train[num], ax=ax[row, col])
    ax[row, col].set_title(f'Distribution of {num}')
    ax[row, col].set_xlabel(num)
    ax[row, col].set_ylabel('Index')

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()


fig, ax = plt.subplots(4, 2, figsize=(20, 15))

fig.suptitle("Distribution of each feature ", fontsize=20)

for i, num in enumerate(num_features.columns):
    row = i // 2
    col = i % 2
    sns.violinplot(x=train[num], ax=ax[row, col])
    ax[row, col].set_title(f'Distribution of {num}')
    ax[row, col].set_xlabel(num)
    ax[row, col].set_ylabel('Index')

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()


categorical_cols = train.select_dtypes(include='object').columns.tolist()

plt.figure(figsize=(15, 12))
for i, col in enumerate(categorical_cols):
    plt.subplot(4, 3, i + 1)
    sns.countplot(data=train, y=col, order=train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.xlabel('Count')
    plt.ylabel(col)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 12))
plt.suptitle("Target Rate by each column")
for i, col in enumerate(categorical_cols):
    plt.subplot(4, 3, i + 1)
    sns.barplot(data=train, x=col, y='y')
    plt.title(f'Target Rate by {col}')
    plt.xlabel(col)
    plt.ylabel('Proportion of y=1')
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


train


columns = [col for col in train.columns if col != 'y']
num_plots = len(columns)

fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(20, 40))
axes = axes.flatten()

for i, col in enumerate(columns):
    grouped = train.groupby(col, as_index=False)['y'].count()
    sns.barplot(x=col, y='y', data=grouped, ax=axes[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count of y')
    axes[i].set_title(f'Count of y by {col}')
    axes[i].tick_params(axis='x', rotation=45)

# Remove any empty subplots if columns < 16
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

