# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train_solution_path = '/kaggle/input/widsdatathon2025/TRAIN/TRAINING_SOLUTIONS.xlsx'
train_categorical = '/kaggle/input/widsdatathon2025/TRAIN/TRAIN_CATEGORICAL_METADATA.xlsx'
train_quantitative = '/kaggle/input/widsdatathon2025/TRAIN/TRAIN_QUANTITATIVE_METADATA.xlsx'
train_functional = '/kaggle/input/widsdatathon2025/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv'
cat = pd.read_excel(train_categorical)
quant = pd.read_excel(train_quantitative)
functional = pd.read_csv(train_functional)
sol = pd.read_excel(train_solution_path)
cat


cat.isna().sum()


cols = cat.columns[1:]
# Generate bar plots for each categorical column
for column in cols:
    cat[column].value_counts().plot(kind='bar')
    plt.title(f'Bar Plot for {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.show()


quant['MRI_Track_Age_at_Scan'].isna().sum()


quant.describe()


for column in quant.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    quant[column].plot(kind='hist', bins=10, edgecolor='black')
    plt.title(f'Histogram for {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.show()



functional


from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
pca_result = pca.fit_transform(functional.iloc[:, 1:])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1])
plt.title('PCA of fMRI Connectome Data')
plt.show()



(functional.shape[1]-1)/199/200*2


[r for r in list(functional.columns) if r.find('199') != -1]


functional.iloc[:, -1]


# merge with solutoin
complete = pd.merge(sol, quant, on='participant_id', how='inner')
complete


corr_matrix = complete.iloc[:, 1:].corr()
corr_matrix


import seaborn as sns

plt.figure(figsize=(15, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=1)
plt.title('Correlation Matrix Heatmap')
plt.show()



cat_encoded = pd.get_dummies(cat, columns=cat.columns[1:])
cat_encoded


# merge with solutions
cat_sol = pd.merge(cat_encoded, sol, on='participant_id', how='inner')
cat_sol


cat_corr = cat_sol.iloc[:, 1:].corr()
cat_corr



plt.figure(figsize=(15, 8))
sns.heatmap(cat_corr, annot=False, cmap='coolwarm', linewidths=1)
plt.title('Correlation Matrix Heatmap')
plt.show()





