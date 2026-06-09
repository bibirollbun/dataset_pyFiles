import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid")


df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col=0)


df_train.shape


df_train.head()


df_train.tail()


df_train.dtypes


df_train.describe()


df_train.describe(include=['object'])


df_train.duplicated().sum()


plt.figure(figsize=(15, 4))
sns.heatmap(df_train.isnull(), yticklabels=False, cbar=False, cmap="Purples")
plt.title("Missing Data Heatmap")
plt.show()


print("Percentage of missing values in each column:")
display(df_train.isnull().sum() / df_train.shape[0] * 100)


plt.figure(figsize=(15, 4))
sns.histplot(data=df_train, x='Time_spent_Alone', kde=True)
plt.title('Distribution of Time spent Alone')
plt.show()


plt.figure(figsize=(15, 8))
sns.boxplot(data=df_train, x='Personality', y='Friends_circle_size')
plt.title('')
plt.show()


plt.figure(figsize=(15, 8))
sns.boxplot(data=df_train, x='Personality', y='Post_frequency')
plt.title('')
plt.show()


g = sns.PairGrid(df_train, hue="Personality")
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()


plt.figure(figsize=(15, 6))
sns.kdeplot(data=df_train)


plt.figure(figsize=(15, 6))
corr = df_train.drop(columns=['Personality']).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


counts1 = df_train['Stage_fear'].value_counts()
counts2 = df_train['Drained_after_socializing'].value_counts()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].pie(counts1, labels=counts1.index, autopct='%1.1f%%', startangle=90)
axs[0].set_title('Stage_fear')

axs[1].pie(counts2, labels=counts2.index, autopct='%1.1f%%', startangle=90)
axs[1].set_title('Drained_after_socializing')

plt.tight_layout()
plt.show()


plt.figure(figsize=[15,10])
sns.countplot(x='Stage_fear', hue='Personality', data=df_train)


plt.figure(figsize=[15,10])
sns.countplot(x='Drained_after_socializing', hue='Personality', data=df_train)




