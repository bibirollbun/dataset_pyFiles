# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
train = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
test = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')


# Quick overview and ensure data loaded properly
print(train.shape)
print(train.head())
print(train['isFraud'].value_counts())


# Target Variable Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train, x='isFraud')
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()


# Distribution of Transaction Amounts
plt.figure(figsize=(10,5))
sns.histplot(train['TransactionAmt'], bins=100, log_scale=(False, True))
plt.title('Transaction Amount Distribution')
plt.xlabel('Transaction Amount')
plt.ylabel('Count (log scale)')
plt.show()


# Categorical Features: card4
plt.figure(figsize=(8,4))
sns.countplot(data=train, x='card4', hue='isFraud')
plt.title('card4 Distribution by Fraud')
plt.show()


# Top 20 features with missing values
missing = train.isnull().mean().sort_values(ascending=False) * 100
missing_df = missing.reset_index()
missing_df.columns = ['Feature', 'Missing Percentage']
missing_df.head(20)


top_missing = missing.head(20)

plt.figure(figsize=(10,6))
sns.barplot(x=top_missing.values, y=top_missing.index, palette='viridis')
plt.xlabel('Missing Value Percentage')
plt.title('Top 20 Features by Missing Data')
plt.tight_layout()
plt.show()

