import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt


train_dataset = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train_dataset.shape


train_dataset.info()


train_dataset.SMILES


train_dataset.describe().T


train_dataset.SMILES.str.startswith('*').all()


!pip install rdkit


from rdkit import Chem
import pandas as pd

def split_aromatic(smiles: str):
    """
    Return a tuple (aromatic_fragment_smiles, non_aromatic_smiles).

    The first element is all atoms that RDKit flags as aromatic, written
    as a dot-separated SMILES if there are several disconnected rings.
    The second element is everything else (stars, hetero-atoms, alkyl chains, …).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return (None, None)

    arom = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    non  = [a.GetIdx() for a in mol.GetAtoms() if not a.GetIsAromatic()]

    arom_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=arom,
                                           kekuleSmiles=False, canonical=True) if arom else ""
    rest_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=non,
                                           kekuleSmiles=False, canonical=True) if non else ""
    return arom_smiles, rest_smiles


# Assuming your DataFrame is called df and has a column “SMILES”
train_dataset[["aromatic_smiles", "non_aromatic_smiles"]] = train_dataset["SMILES"].apply(
    lambda s: pd.Series(split_aromatic(s))
)



train_dataset




