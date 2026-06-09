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



!pip install missingno rdkit-pypi py3Dmol -q


import numpy
numpy.__version__


# import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msgo

# rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import py3Dmol


import warnings
warnings.filterwarnings('ignore')


# load the data
sample_submission_df = pd.read_csv(r"/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")
train_df = pd.read_csv(r"/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test_df = pd.read_csv(r"/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")


# see data
train_df.head()


# let's understand each column one by one
train_df.info()


print(f"`id` is the unique {train_df['id'].nunique()} == {train_df.shape[0]}")


train_df['SMILES']


train_df['Tg']


train_df['FFV']


train_df['Tc']


train_df['Density']


train_df['Tg']


msgo.matrix(train_df)


train_df.isnull().sum()


# percentage of data is missing
(train_df.isnull().sum() / train_df.shape[0]) * 100 


train_df[train_df.duplicated()]


corr = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].corr()
mask = np.triu(np.ones_like(corr))
sns.heatmap(corr, annot=True, cmap='coolwarm', mask=mask, center=0)
plt.title("Correlation Between Target Properties")
plt.show()



corr


g = sns.PairGrid(corr)
g.map_diag(sns.histplot)
g.map_lower(sns.scatterplot)
g.map_upper(sns.kdeplot)


train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].hist(bins=25, figsize=(15, 8))
plt.show()


train_df.columns


for y in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']: 
    sns.scatterplot(
        data=train_df,
        x=train_df['SMILES'].apply(len),
        y=y,
    )
    plt.show()


sns.boxplot(
    x= train_df['SMILES'].apply(len),
    y=pd.qcut(train_df['Tg'], q=10),
    
)
plt.show()


train_df['SMILES'].iloc[1]


def smiles_details():
    """Randomly generate the 2d molecule structure & properties."""
    smiles = np.random.choice(train_df['SMILES'].to_numpy())    
    mol = Chem.MolFromSmiles(smiles)
    print(smiles)
    
    # descriptors
    descriptors = {
        "Molecule Weight" : Descriptors.MolWt(mol),
        "No of H-bond Acceptors": Descriptors.NumHAcceptors(mol),
        "Num of H-bond Donors": Descriptors.NumHDonors(mol),
        "Num of Rings": Descriptors.RingCount(mol),
        "Num of Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Heavy Atom Count": Descriptors.HeavyAtomMolWt(mol),
        "Topological Polar Surface Area (TPSA)": Descriptors.TPSA(mol),
        "LogP (Hydrophobicity)": Descriptors.MolLogP(mol),
        "Num of Aromatic Rings": Descriptors.NumAromaticRings(mol),
        "Num of Aromatic Atoms": Descriptors.NumHeteroatoms(mol)
    }
    
    df = pd.DataFrame(list(descriptors.items()), columns=["Descriptors", "Value"])
    print(df)
    
    return Draw.MolToImage(mol)


smiles_details()


def show_3d_smiles():
    """Randomly generate the 3d molecule structure."""
    smiles = np.random.choice(train_df['SMILES'].to_numpy())    
    mol = Chem.MolFromSmiles(smiles)
    print(smiles)
    
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    block = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=800, height=400)
    view.addModel(block, "mol")
    view.setStyle({'stick': {}})
    view.zoomTo()
    view.show()



show_3d_smiles()

