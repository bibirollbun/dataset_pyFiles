import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')


train = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


print('Train shape:', train.shape)
print('Test shape :', test.shape)


train.head()


train.info()


train['diagnosed_diabetes'].value_counts(normalize=True)


sns.countplot(x='diagnosed_diabetes', data=train)
plt.title('Target Distribution: Diagnosed Diabetes')
plt.show()


train.isna().sum()


binary_features = [col for col in train.columns if train[col].nunique() == 2 and col != 'diagnosed_diabetes']
continuous_features = [col for col in train.columns if train[col].nunique() > 10]


print('Binary features:', len(binary_features))
print('Continuous features:', len(continuous_features))


for col in continuous_features[:6]:
    plt.figure(figsize=(5,3))
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()


for col in binary_features[:6]:
    sns.countplot(x=col, data=train)
    plt.title(f'{col} Distribution')
    plt.show()


for col in continuous_features[:5]:
    sns.boxplot(x='diagnosed_diabetes', y=col, data=train)
    plt.title(f'{col} vs Diabetes Outcome')
    plt.show()


for col in binary_features[:5]:
    pd.crosstab(train[col], train['diagnosed_diabetes'], normalize='index').plot(kind='bar', stacked=True)
    plt.title(f'{col} vs Diabetes Outcome')
    plt.ylabel('Proportion')
    plt.show()


plt.figure(figsize=(12,8))
corr = train[continuous_features + ['diagnosed_diabetes']].corr()
sns.heatmap(corr, cmap='coolwarm',annot = True,center=0)
plt.title('Correlation Heatmap')
plt.show()


for col in continuous_features[:4]:
    sns.kdeplot(train[col], label='Train')
    sns.kdeplot(test[col], label='Test')
    plt.title(f'Train vs Test Distribution: {col}')
    plt.legend()
    plt.show()

