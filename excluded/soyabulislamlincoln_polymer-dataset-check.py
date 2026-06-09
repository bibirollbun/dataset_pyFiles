import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import *


train_data =pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train_data.head()


train_data.info()


train_data.columns


test_data = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
test_data.head()


sample.head()


!pip install rdkit  mbuild openmm


from openmm.app import ForceField
from openmm.unit import *
from rdkit import Chem
from rdkit.Chem import *
from matplotlib.colors import ColorConverter


train_data['SMILES'][0]


molecule = Chem.MolFromSmiles(train_data['SMILES'][0])
molecule



img = Draw.MolToImage(molecule , highlightColor=ColorConverter().to_rgb('aqua'))

plt.imshow(img)


molecule2 = Chem.MolFromSmiles(train_data['SMILES'][1])
img2 = Draw.MolToImage(molecule2 ,size=(224, 224), highlightColor=ColorConverter().to_rgb('aqua'))

plt.imshow(img2)


molecule2 = Chem.MolFromSmiles(train_data['SMILES'][4])
img2 = Draw.MolToImage(molecule2 , highlightColor=ColorConverter().to_rgb('aqua'))

plt.imshow(img2)


train_data['SMILES'][4]


def smiles_to_3d_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.AddHs(mol)  
    AllChem.EmbedMolecule(mol, randomSeed=42) 
    AllChem.MMFFOptimizeMolecule(mol)  
    return mol



import py3Dmol
from rdkit.Chem import AllChem

molecule_3d = smiles_to_3d_structure(train_data['SMILES'][7])

Chem.Draw.IPythonConsole.ipython_3d = True

# Draw the molecule in 3D
view = Chem.Draw.IPythonConsole.drawMol3D(molecule_3d, size=(700, 700))

view








