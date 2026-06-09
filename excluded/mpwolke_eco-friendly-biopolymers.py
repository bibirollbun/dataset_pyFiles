# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
submission.tail()


test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test.tail()


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train.tail()


train.info()


# Check if there are any missing values left
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
missing_data.head()


#Describe showing Only the requested statistics (mean, minimum and maximum). Then, transpose the table.

train.describe().loc[['mean','min','max']].T


numerical_cols = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']


train[numerical_cols].hist(figsize=(15,10), bins=30, color='Green', edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.show()


# OutlierPandas https://www.kaggle.com/code/abhyudaya456/s5e6-eda-for-predicting-optimal-fertilizers/notebook 
plt.figure(figsize=(10,6))
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='summer')
plt.title("Correlation Between Numerical Features")
plt.show()


#This is the cheminformatics package that will do most of the heavy lifting for us
!pip install rdkit


#By Chemdatafarmer  https://www.kaggle.com/code/chemdatafarmer/additional-seh-data/notebook
#By Meer Atif https://www.kaggle.com/code/meeratif/smiles-open-problems

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger


#https://www.rdkit.org/docs/source/rdkit.Chem.Draw.html

#By Chemdatafarmer  https://www.kaggle.com/code/chemdatafarmer/additional-seh-data/notebook

###Visualize some of the original data
Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in train['SMILES'][0:4]], molsPerRow=4, subImgSize=(400,300))


#https://greglandrum.github.io/rdkit-blog/posts/2023-10-25-molsmatrixtogridimage.html
#https://www.rdkit.org/docs/source/rdkit.Chem.Draw.html
#By Chemdatafarmer  https://www.kaggle.com/code/chemdatafarmer/additional-seh-data/notebook

###Visualize some of the original data
Draw.MolsToGridImage([Chem.MolFromSmiles(x) for x in train['SMILES'][0:12]], molsPerRow=3, subImgSize=(200,200))


train.iloc[1,1]


#Row 7969

train.iloc[7969,1]


#Row 7972nd

train.iloc[7972, 1]


!pip install pysmiles


#https://mattermodeling.stackexchange.com/questions/6460/rdkit-and-pysmiles-results-differ-on-some-smiles-strings
#Answered by RapelPy, Jul 31, 2021

from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import pysmiles

s1 = '*Nc1ccc([C@H](CCC)c2ccc(C3(c4ccc([C@@H](CCC)c5ccc(N*)cc5)cc4)CCC(CCCCC)CC3)cc2)cc1'  # aromatic
s2 = '*c1ccc(OCCCCCCCCCCCOC(=O)CCCCC(=O)OCCCCCCCCCCCOc2ccc(-c3nnc(*)s3)cc2)cc1'  # kekulized


def show_implicit_h(smiles):
    m = Chem.MolFromSmiles(smiles)
    for atom in m.GetAtoms():
        atom.SetProp('atomLabel', str(atom.GetIdx()))
    m = Chem.AddHs(m)
    return Draw.MolToImage(m, size=(300, 300))


show_implicit_h(s1)


#https://mattermodeling.stackexchange.com/questions/6460/rdkit-and-pysmiles-results-differ-on-some-smiles-strings

show_implicit_h(s2)


from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from IPython.display import SVG
IPythonConsole.ipython_useSVG=True  


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


mol = Chem.MolFromSmiles("*Oc1cc(CCCCCCCC)cc(OC(=O)c2cccc(C(*)=O)c2)c1") #7968 row
mol = mol_with_atom_index(mol)
mc = Chem.Mol(mol.ToBinary())

drawer = rdMolDraw2D.MolDraw2DSVG(450, 200) 
drawer.DrawMolecule(mc)
drawer.FinishDrawing()

svg = drawer.GetDrawingText()
display(SVG(svg.replace('svg:','')))


%matplotlib inline

sns.set_style('white')

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw, PyMol, rdFMCS
from rdkit.Chem.Draw import IPythonConsole
from rdkit import rdBase


#https://deepchem.io/tutorials/creating-a-high-fidelity-model-from-experimental-data/

train['len'] = [len(i) if i is not None else 0 for i in train['SMILES']]
smiles_lens = [len(i) if i is not None else 0 for i in train['SMILES']]
sns.histplot(smiles_lens)
plt.xlabel('len(smiles)')
plt.ylabel('probability')
plt.title('Smiles Len');


#https://deepchem.io/tutorials/creating-a-high-fidelity-model-from-experimental-data/

# indices of large looking molecules
suspiciously_large = np.where(np.array(smiles_lens) > 150)[0]

# corresponding smiles string
long_smiles = train.loc[train.index[suspiciously_large]]['SMILES'].values

# look
Draw._MolsToGridImage([Chem.MolFromSmiles(i) for i in long_smiles], molsPerRow=6)

