!pip install rdkit


import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go


from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Draw
from rdkit.Chem import PandasTools, AllChem, rdFingerprintGenerator

from sklearn.decomposition import PCA

import warnings

warnings.filterwarnings('ignore')

sns.set_style("white")
%matplotlib inline


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv", index_col=0)
df_test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv", index_col=0)


df_train.head()


df_test.head()


print("Training set size: ", df_train.shape)
print("Testing set size: ", df_test.shape)


print("Number of missing values in each column:")
display(df_train.isnull().sum())
print()
print("Percentage of missing values in each column:")
display(df_train.isnull().sum() / df_train.shape[0] * 100)


df_train['mol'] = df_train['SMILES'].apply(Chem.MolFromSmiles)
df_train = df_train[df_train['mol'].notna()] 


Draw.MolsToGridImage(df_train['mol'].sample(9).tolist(), molsPerRow=3, subImgSize=(500, 500))


df_train['MolWt'] = df_train['mol'].apply(Descriptors.MolWt)
df_train['LogP'] = df_train['mol'].apply(Crippen.MolLogP)
df_train['NumHDonors'] = df_train['mol'].apply(Lipinski.NumHDonors)
df_train['NumHAcceptors'] = df_train['mol'].apply(Lipinski.NumHAcceptors)
df_train['TPSA'] = df_train['mol'].apply(Descriptors.TPSA)
df_train['NumRotatableBonds'] = df_train['mol'].apply(Descriptors.NumRotatableBonds)
df_train['RingCount'] = df_train['mol'].apply(rdMolDescriptors.CalcNumRings)
df_train['HeavyAtomCount'] = df_train['mol'].apply(Descriptors.HeavyAtomCount)


df_train.describe()


plt.figure(figsize=(10, 8))
df_corr = df_train[['MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 'RingCount']].corr()

ax = plt.matshow(df_corr, cmap='coolwarm', fignum=1)
plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=90)
plt.yticks(range(len(df_corr.columns)), df_corr.columns)
plt.colorbar()

ax.axes.xaxis.set_ticks_position('bottom')
ax.axes.xaxis.set_label_position('bottom')

plt.title("Descriptor Correlation Matrix", y=1.15)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(16, 12))  
features = ['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'HeavyAtomCount']

for ax, feat in zip(axes.flat, features):
    sns.histplot(df_train[feat], ax=ax, kde=True)
    ax.set_title(feat)

fig.suptitle("Distribution of molecular descriptors", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(3, 2, figsize=(16, 8))
features = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
for ax, feat in zip(axes.flat, features):
    sns.histplot(df_train[feat], ax=ax, kde=True)
    ax.set_title(feat)
fig.suptitle("Distribution of target properties", y=1.02, fontsize=16)    
plt.tight_layout()
plt.show()


df_numeric = df_train[['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'HeavyAtomCount', "Tg"]]
df_numeric.corr()['Tg'].sort_values()


sns.regplot(data=df_train, x="RingCount", y="Tg")
plt.ylim(0,)


df_numeric = df_train[['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'HeavyAtomCount', "Tc"]]
df_numeric.corr()['Tc'].sort_values()


sns.regplot(data=df_train, x="NumRotatableBonds", y="Tc")
plt.ylim(0,)


sns.regplot(data=df_train, x="LogP", y="Tc")
plt.ylim(0,)


df_numeric = df_train[['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'HeavyAtomCount', "Density"]]
df_numeric.corr()['Density'].sort_values()


sns.regplot(data=df_train, x="NumHAcceptors", y="Density")
plt.ylim(0,)


sns.regplot(data=df_train, x="NumRotatableBonds", y="Density")
plt.ylim(0,)


df_numeric = df_train[['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'HeavyAtomCount', "Rg"]]
df_numeric.corr()['Rg'].sort_values()


sns.regplot(data=df_train, x="NumRotatableBonds", y="Rg")
plt.ylim(0,)


sns.regplot(data=df_train, x="MolWt", y="Rg")
plt.ylim(0,)


df_numeric = df_train[['MolWt', 'LogP', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds', 'RingCount', 'HeavyAtomCount', "FFV"]]
df_numeric.corr()['FFV'].sort_values()


sns.regplot(data=df_train, x="LogP", y="FFV")
plt.ylim(0,)


sns.regplot(data=df_train, x="NumHDonors", y="FFV")
plt.ylim(0,)

