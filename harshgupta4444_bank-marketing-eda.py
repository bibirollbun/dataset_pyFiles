import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud


df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
print(f"Shape: {df.shape}")
df.head()


plt.figure(figsize=(6,4))
sns.countplot(data=df, x='y')
plt.title("Target Variable Distribution (0: No, 1: Yes)")
plt.show()


num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
df[num_cols].describe()


fig, axes = plt.subplots(2, 2, figsize=(12,8))
sns.histplot(df['age'], bins=30, kde=True, ax=axes[0,0])
sns.boxplot(df['balance'], ax=axes[0,1])
sns.histplot(df['duration'], bins=50, kde=True, ax=axes[1,0])
sns.boxplot(df['campaign'], ax=axes[1,1])
plt.tight_layout()


plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols + ['y']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")


cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for col in cat_cols:
    plt.figure(figsize=(10,4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"Distribution of {col}")
    plt.show()


for col in cat_cols:
    plt.figure(figsize=(10,4))
    sns.barplot(data=df, x=col, y='y', order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"Subscription Rate by {col}")
    plt.show()


for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df, x='y', y=col)
    plt.title(f"{col} vs Subscription")
    plt.show()


pd.crosstab(df['job'], df['education'], values=df['y'], aggfunc='mean')




