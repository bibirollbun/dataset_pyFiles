import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", )
sns.set(style="whitegrid")


ROOT_DATA_DIR = Path("/kaggle/input/playground-series-s5e8/")
train_df = pd.read_csv(os.path.join(ROOT_DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(ROOT_DATA_DIR, 'test.csv'))

print(train_df.head())
print(train_df.info())


plt.figure(figsize=(10, 6))
sns.histplot(train_df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 8))
job_counts = train_df['job'].value_counts()
plt.pie(job_counts, labels=job_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Job Type Distribution')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='marital')
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='education')
plt.title('Education Level Distribution')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(train_df['balance']), bins=50, kde=True)
plt.title('Log Balance Distribution')
plt.xlabel('Log(Balance + 1)')
plt.ylabel('Frequency')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=train_df, x='housing', ax=ax[0])
ax[0].set_title('Housing Loan Distribution')
sns.countplot(data=train_df, x='loan', ax=ax[1])
ax[1].set_title('Personal Loan Distribution')
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(data=train_df, x='y')
plt.title('Target Variable Distribution (y)')
plt.xlabel('Subscribe')
plt.ylabel('Count')
plt.show()

print(train_df['y'].value_counts(normalize=True))


plt.figure(figsize=(12, 6))
sns.countplot(data=train_df, x='job', hue='y')
plt.title('Job vs Subscription')
plt.xticks(rotation=45)
plt.legend(title='Subscribe')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=train_df, x='y', y='age')
plt.title('Age vs Subscription')
plt.xlabel('Subscribe')
plt.ylabel('Age')
plt.show()

numeric_cols = ['age', 'balance', 'day']
if 'duration' in train_df.columns:
    numeric_cols.append('duration')

corr = train_df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()




