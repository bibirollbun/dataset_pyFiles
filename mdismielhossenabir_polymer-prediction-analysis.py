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


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train


test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test


sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
sub


train.info()


train.isnull().sum()


import matplotlib.pyplot as plt

train.isnull().mean().plot(kind="bar", figsize=(10,6), title="Missing Values Fraction")
plt.show()


for col in ['Tg','FFV','Tc','Density','Rg']:
    train[col].fillna(train[col].mean(), inplace=True)


train.isnull().sum()


import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(train[['Tg','FFV','Tc','Density','Rg']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()



!pip install rdkit-pypi


from rdkit import Chem
from rdkit.Chem import Descriptors

train["MolWt"] = train["SMILES"].dropna().apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
train["MolWt"].hist(bins=30, figsize=(6,4))
plt.title("Distribution of Molecular Weight")
plt.xlabel("MolWt")
plt.ylabel("Count")
plt.show()


top = train.nlargest(10, "MolWt")[["id","SMILES","MolWt"]]
top


train['SMILES'].head()


from rdkit.Chem import Draw

smiles = "*CC(*)c1ccccc1C(=O)OCCCCCC"  
mol = Chem.MolFromSmiles(smiles)

Draw.MolToImage(mol)


smiles_list = train["SMILES"].dropna().head(10)
mols = [Chem.MolFromSmiles(s) for s in smiles_list]

img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300,300))
img


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def molwt(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan
        return Descriptors.MolWt(mol)
    except:
        return np.nan


train["MolWt"] = train["SMILES"].apply(molwt)
test["MolWt"] = test["SMILES"].apply(molwt)

train_molwt_mean = train["MolWt"].mean()
test_molwt_mean = test["MolWt"].mean()

train["MolWt"].fillna(train_molwt_mean, inplace=True)
test["MolWt"].fillna(test_molwt_mean, inplace=True)


X = ["MolWt"] 
y = ["Tg", "FFV", "Tc", "Density", "Rg"]


valid_rows = train[y].notna().any(axis=1)
X_train = train.loc[valid_rows, X]
y_train = train.loc[valid_rows, y]


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


X_test = test[X]

test_pred = model.predict(X_test)


for i, col in enumerate(y):
    rmse = mean_squared_error(y_val[col], y_pred_val[:, i], squared=False)
    r2 = r2_score(y_val[col], y_pred_val[:, i])
    print(f"{col}: RMSE = {rmse:.4f}, R2 = {r2:.4f}")

