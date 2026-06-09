import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


train.info()


train.head()


train.shape


train.isnull().sum()


for column in train.columns:
    unique_values = train[column].unique()
    print(f"Column: {column}")
    print(f"Unique Values ({len(unique_values)}): {unique_values}")
    print("-" * 50)


print("\nSummary Statistics:\n", train.describe())


test.info()


test.head()


test.shape


test.isnull().sum()


for column in test.columns:
    unique_values = test[column].unique()
    print(f"Column: {column}")
    print(f"Unique Values ({len(unique_values)}): {unique_values}")
    print("-" * 50)


print("\nSummary Statistics:\n", test.describe())


numerical_cols = [col for col in train.columns if col in test.columns and train[col].dtype in ['float64', 'int64']]
numerical_cols


fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(10, 4 * len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    sns.histplot(train[col], bins=30, edgecolor='black', ax=axes[i][0])
    axes[i][0].set_title(f"Train {col}")

    sns.histplot(test[col], bins=30, edgecolor='black', ax=axes[i][1])
    axes[i][1].set_title(f"Test {col}")

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(10, 4 * len(numerical_cols)))

for i, col in enumerate(numerical_cols):
    sns.boxplot(x=train[col], ax=axes[i][0])
    axes[i][0].set_title(f"Train {col}")
    
    sns.boxplot(x=test[col], ax=axes[i][1])
    axes[i][1].set_title(f"Test {col}")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
sns.heatmap(train.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()


features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
            'Guest_Popularity_percentage', 'Number_of_Ads']

plt.figure(figsize=(16, 12))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.scatterplot(data=train, x=feature, y='Listening_Time_minutes', alpha=0.5)
    plt.title(f'Listening Time vs {feature}')
    plt.tight_layout()

plt.show()


train_copy = train.copy().assign(Dataset='Train')
test_copy = test.copy().assign(Dataset='Test', Listening_Time_minutes=None)

combined = pd.concat([train_copy, test_copy])


num_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
            'Guest_Popularity_percentage', 'Number_of_Ads']

plt.figure(figsize=(16, 12))

for i, col in enumerate(num_cols, 1):
    plt.subplot(2, 2, i)
    sns.kdeplot(data=combined, x=col, hue='Dataset', alpha=0.5)
    plt.title(f'{col} Distribution: Train vs Test')
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

