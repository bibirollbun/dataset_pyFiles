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
import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample of the training data
usecols = ['customer_ID', 'S_2']
train = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', usecols=usecols, nrows=200000)
labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')

train['S_2'] = pd.to_datetime(train['S_2'])
latest = train.sort_values(['customer_ID','S_2']).groupby('customer_ID').tail(1)
merged = latest.merge(labels, on='customer_ID', how='left')

# Default rate by month
merged['month'] = merged['S_2'].dt.to_period('M').astype(str)
trend = merged.groupby('month')['target'].mean().reset_index()

# Visualization
plt.figure(figsize=(10,5))
sns.lineplot(data=trend, x='month', y='target', marker='o', color='orange')
plt.title('Average Default Rate Over Time')
plt.xlabel('Month')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Load relevant columns
usecols = ['customer_ID', 'S_2', 'P_2']  # P_2 = total payment amount
train = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', usecols=usecols, nrows=200000)
labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')

train['S_2'] = pd.to_datetime(train['S_2'])
latest = train.sort_values(['customer_ID','S_2']).groupby('customer_ID').tail(1)
merged = latest.merge(labels, on='customer_ID', how='left')

# Bucket payments
merged['payment_bucket'] = pd.qcut(merged['P_2'], q=5, labels=['Very Low','Low','Medium','High','Very High'])

# Default rate by payment bucket
payment_default = merged.groupby('payment_bucket')['target'].mean().reset_index()

# Visualization
plt.figure(figsize=(8,5))
sns.barplot(data=payment_default, x='payment_bucket', y='target', palette='coolwarm')
plt.title('Default Rate by Payment Amount Category')
plt.xlabel('Payment Level')
plt.ylabel('Default Rate')
plt.tight_layout()
plt.show()



# Load sample of spending and balance columns
usecols = ['customer_ID', 'S_2', 'S_3', 'B_1']  # S_3=spend amount, B_1=balance
train = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', usecols=usecols, nrows=200000)
labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')

train['S_2'] = pd.to_datetime(train['S_2'])
latest = train.sort_values(['customer_ID','S_2']).groupby('customer_ID').tail(1)
merged = latest.merge(labels, on='customer_ID', how='left')

# Scatter plot: Spend vs Balance, colored by default
plt.figure(figsize=(8,6))
sns.scatterplot(data=merged.sample(5000), x='B_1', y='S_3', hue='target', palette='Set1', alpha=0.6)
plt.title('Spend vs Balance — Colored by Default Status')
plt.xlabel('Balance (B_1)')
plt.ylabel('Spend (S_3)')
plt.legend(title='Default', labels=['No Default','Default'])
plt.tight_layout()
plt.show()



import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

# Load relevant columns
usecols = ['customer_ID','S_2','D_39']
train = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', usecols=usecols, nrows=200000)
labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')

train['S_2'] = pd.to_datetime(train['S_2'])
latest = train.sort_values(['customer_ID','S_2']).groupby('customer_ID').tail(1)
merged = latest.merge(labels, on='customer_ID', how='left')

# Bucket delinquency score
merged['delinquency_bucket'] = pd.qcut(merged['D_39'], q=5, labels=['Very Low','Low','Medium','High','Very High'])

# Default rate by delinquency bucket
delinq_rate = merged.groupby('delinquency_bucket')['target'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=delinq_rate, x='delinquency_bucket', y='target', palette='rocket')
plt.title('Default Rate by Delinquency Level (D_39)')
plt.xlabel('Delinquency Level')
plt.ylabel('Default Rate')
plt.tight_layout()
plt.show()



usecols = ['customer_ID','S_2','S_3']
train = pd.read_csv('/kaggle/input/amex-default-prediction/train_data.csv', usecols=usecols, nrows=150000)
labels = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')

train['S_2'] = pd.to_datetime(train['S_2'])
latest = train.sort_values(['customer_ID','S_2']).groupby('customer_ID').tail(1)
merged = latest.merge(labels, on='customer_ID', how='left')

plt.figure(figsize=(8,5))
sns.kdeplot(data=merged, x='S_3', hue='target', fill=True, common_norm=False, palette='muted')
plt.title('Spending Distribution (S_3) by Default Status')
plt.xlabel('Spending Amount')
plt.ylabel('Density')
plt.tight_layout()
plt.show()


