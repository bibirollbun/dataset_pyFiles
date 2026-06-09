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


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
train.set_index("id", inplace = True)

train.drop_duplicates(inplace = True)
train.head()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




for col in train.columns:

    if train[col].dtype == 'object':
        plt.figure(figsize=(8, 6))
        sns.countplot(x=col, data=train, color = "blue")
        plt.title(f'Frequency Distribution of {col}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.histplot(train[col], bins=20, kde=True, color='blue', ax=ax1)
        ax1.set_title(f'Frequency Distribution of {col}')
            

        sns.boxplot(y=train[col], ax=ax2, palette='Set2')
        ax2.set_title(f'{col} Distribution')
        
        plt.tight_layout()
        plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

numerical_columns = train.select_dtypes(include=['number']).columns


bin_increment = 5


for col in numerical_columns:
    if col != "Calories": 
        avg_feature = train.groupby(col)["Calories"].mean()

        max_avg_feature = avg_feature.max()  
        bins = range(0, int(max_avg_feature) + bin_increment, bin_increment)

        avg_feature_binned = pd.cut(avg_feature, bins=bins, right=False, labels=False)


        binned_avg_feature = avg_feature.groupby(avg_feature_binned).mean()


        fig, ax1 = plt.subplots(figsize=(15, 5))

        x = range(len(binned_avg_feature))
        bar_width = 1.0  
        bars = ax1.bar(x, binned_avg_feature.values, width=bar_width, color='blue', edgecolor='black')

        ax1.plot(x, binned_avg_feature.values, color='red', marker='o', linestyle='-', linewidth=2, label='Trend')


        bin_labels = [f'({bins[i]}, {bins[i+1]})' for i in range(len(binned_avg_feature))]

        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax1.set_xlabel(f"{col}")
        ax1.set_ylabel(f'Average Calories')
        ax1.set_title(f'Average Calories vs {col} (Binned Averages)')
        ax1.legend()
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()



#Replace male and female with 0 and 1. 

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})



import itertools

cross_cols = train.columns[~train.columns.isin(['Calories', 'Sex'])]

for col1, col2 in itertools.combinations(cross_cols, 2):

    train[f'{col1}_minus_{col2}'] = train[col1] - train[col2]
    train[f'{col1}_times_{col2}'] = train[col1] * train[col2]

train.head()



correlation_matrix = train.corr()


plt.figure(figsize=(15, 12))


sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, square=True)


plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)


plt.title('Correlation Matrix of Features', fontsize=16)
plt.tight_layout()


plt.show()




numerical_columns = [col for col in train.columns if col != "Calories"]


for i in range(0, len(numerical_columns), 4):
    cols = numerical_columns[i:i+4]
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5))  # Dynamic sizing


    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sns.scatterplot(x=train[col], y=train["Calories"], ax=ax, color='blue', s=50, edgecolor='black')
        ax.set_xlabel(col)
        ax.set_ylabel("Calories")
        ax.set_title(f"{col} vs Calories")
        ax.grid(True)

    plt.tight_layout()
    plt.show()





