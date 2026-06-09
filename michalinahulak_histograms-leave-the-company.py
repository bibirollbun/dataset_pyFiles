import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/machine-learning-4-sbu/train.csv')
test = pd.read_csv('/kaggle/input/machine-learning-4-sbu/test.csv')


def plot_compare(train, test, col, kind='numeric'):
    plt.figure(figsize=(12, 4))
    
    if kind == 'numeric':
        plt.subplot(1, 2, 1)
        sns.histplot(train[col], kde=True, color='blue')
        plt.title(f'Train - {col}')
        
        plt.subplot(1, 2, 2)
        sns.histplot(test[col], kde=True, color='orange')
        plt.title(f'Test - {col}')
        
    elif kind == 'categorical':
        plt.subplot(1, 2, 1)
        train[col].value_counts(normalize=True).plot(kind='bar', color='blue')
        plt.title(f'Train - {col}')
        
        plt.subplot(1, 2, 2)
        test[col].value_counts(normalize=True).plot(kind='bar', color='orange')
        plt.title(f'Test - {col}')
        
    plt.tight_layout()
    plt.show()


target = 'left_company'

plt.figure(figsize=(6,4))
sns.countplot(x=target, data=train, palette='Greens')
plt.title('Target distribution (train only)')
plt.xlabel('Left Company (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()


numeric_cols = train.select_dtypes(include='number').columns.drop(target)
categorical_cols = train.select_dtypes(include='object').columns.intersection(test.columns)


for col in list(numeric_cols):
    plot_compare(train, test, col, kind='numeric')

for col in list(categorical_cols):
    plot_compare(train, test, col, kind='categorical')

