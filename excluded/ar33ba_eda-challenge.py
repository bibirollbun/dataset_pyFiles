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


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import optuna
from sklearn.model_selection import StratifiedKFold
from lightgbm import early_stopping, log_evaluation


# --- Load Datasets ---
print(" Loading datasets...")

train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

# --- Merge transaction + identity ---
print(" Merging transaction and identity datasets...")
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
test = test_transaction.merge(test_identity, on='TransactionID', how='left')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# --- Basic EDA: Missing Values Before Preprocessing ---
print("\n Checking Missing Values before preprocessing...")
missing = train.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0]
plt.figure(figsize=(10, 6))
missing.head(30).plot(kind='barh')
plt.title("Top Missing Values (Before Cleaning)")
plt.show()

# --- Data Type Overview ---
print("\n Data types BEFORE preprocessing:")
print(train.dtypes.value_counts())
print(test.dtypes.value_counts())

# --- Rename columns: '-' to '_' ---
print("\n Replacing '-' with '_' in test columns...")
test.columns = test.columns.str.replace('-', '_', regex=False)




# --- Set random seed ---
SEED = 42
np.random.seed(SEED)

# --- EDA Functions ---

def plot_class_distribution(train, target_column):
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_column, data=train)
    plt.title('Fraud vs Non-Fraud')
    plt.xlabel('Is Fraud (1=Yes, 0=No)')
    plt.ylabel('Count')
    plt.show()


def plot_feature_distributions(train, features, target_column):
    for feature in features:
        plt.figure(figsize=(10, 4))
        sns.kdeplot(train[train[target_column]==0][feature], label='Non-Fraud', fill=True)
        sns.kdeplot(train[train[target_column]==1][feature], label='Fraud', fill=True)
        plt.title(f'{feature} distribution by Fraud/Non-Fraud')
        plt.xlabel(feature)
        plt.legend()
        plt.show()

def plot_correlation(train, features, target_column):
    selected = train[features + [target_column]]
    corr = selected.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title('Correlation Heatmap')
    plt.show()

# --- EDA Execution ---

# Assume train and test are already loaded and preprocessed from your given code

target = 'isFraud'

print("\n Running EDA...")
plot_class_distribution(train, target)

important_features = ['TransactionAmt', ]
plot_feature_distributions(train, important_features, target)


def plot_card_distributions(train, features):
    for feature in features:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=feature, data=train)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

# Usage
card_features = ['card4', 'card6']
plot_card_distributions(train, card_features)


import seaborn as sns
import matplotlib.pyplot as plt
train['card4'] = train['card4'].fillna('Unknown')
train['card6'] = train['card6'].fillna('Unknown')

def plot_card_vs_fraud(train, feature):
    plt.figure(figsize=(8,6))
    ax = sns.countplot(data=train, x=feature, hue='isFraud',
                       order=train[feature].value_counts().index)
    plt.title(f'{feature} vs Fraud')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Add counts on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)

    plt.legend(title='Fraud')
    plt.show()

# Usage
plot_card_vs_fraud(train, 'card4')
plot_card_vs_fraud(train, 'card6')



plt.figure(figsize=(8,4))
sns.countplot(x='DeviceType', hue='isFraud', data=train)
plt.title('DeviceType by Fraud Status', fontsize=14)
plt.xlabel('Device Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()



plt.figure(figsize=(8,4))
sns.countplot(x='ProductCD', hue='isFraud', data=train)
plt.title('Product Code by Fraud Status', fontsize=14)
plt.xlabel('Product Code', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()







