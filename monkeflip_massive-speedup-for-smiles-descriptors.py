from mordred import Calculator, descriptors
from joblib import Parallel, delayed

from rdkit import Chem

import pandas as pd


calc = Calculator(descriptors, ignore_3D=True)

# Function for processing one SMILES (basic example)
def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        desc = calc(mol)
        return desc
    except Exception as e:
        print(f"Error for SMILES {smiles}: {e}")
        return None


# === DATA (head100 as example) ===
train_df = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv").head(100)


# === DESCRIPTORS ===
# kaggle gives you CPU with 4 cores, so 4x faster (locally it depends on your CPU)
desc_train_list = Parallel(n_jobs=-1, verbose=10)(
    delayed(smiles_to_descriptors)(smi) for smi in train_df['SMILES']
)
desc_train_list = [d for d in desc_train_list if d is not None]
desc_train_df = pd.DataFrame([d.asdict() for d in desc_train_list])

