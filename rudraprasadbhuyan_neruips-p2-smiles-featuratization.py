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


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl -q


import numpy as np
import pandas as pd
from tqdm import tqdm 
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import warnings

warnings.filterwarnings('ignore')


# load the data
sum_path = "/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv"
train_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
test_path = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'

sample_submission_df = pd.read_csv(sum_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)


# this give us both name and function 
Descriptors.descList


desc_names = [desc[0] for desc in Descriptors.descList]

def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(desc_names)
    
    values = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            if not np.isfinite(val) or abs(val) > 1e10:
                val = np.nan
        except Exception:
            val = np.nan
        values.append(val)
    return values


descriptors = []
for smi in tqdm(train_df['SMILES'].to_list(), desc="Computing Descriptors :"):
    descriptors.append(compute_all_descriptors(smi))

desc_df = pd.DataFrame(descriptors, columns=desc_names)


desc_df.shape


desc_df


desc_df.describe()


desc_df.columns


def generate_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    fingerprints = []
    
    for smi in tqdm(smiles_list, desc="Generating Morgan Fingerprints :"):
        mol = Chem.MolFromSmiles(smi)
        if  mol is None:
            fingerprints.append([0]*n_bits)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
    
    fp_df = pd.DataFrame(fingerprints, columns=[f"fp_{i}" for i in range(n_bits)])
    return fp_df
    


from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Create generator once (faster)
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

def mol_to_morgan_fp(mol):
    return list(morgan_gen.GetFingerprint(mol))



train_df['mol'] = train_df['SMILES'].apply(Chem.MolFromSmiles)
fp_array = np.array([mol_to_morgan_fp(m) for m in train_df['mol']])
fp_df = pd.DataFrame(fp_array, columns=[f"FP_{i}" for i in range(fp_array.shape[1])])


fp_df


print(f"Descriptors Data Frame Shape :{desc_df.shape} ")
print(f"Morgan Finger Print :{fp_df.shape}")
print(f"Train Data Frame :{train_df.shape}")


train_df.drop(columns=['mol'], inplace=True)


def combine_features(train_df, desc_df, fp_df, drop_cols=['SMILES']):
    combined = pd.concat([train_df.reset_index(drop=True), desc_df, fp_df], axis=1)
    combined = combined.drop(columns=drop_cols, errors='ignore')
    return combined


X_train = combine_features(train_df=train_df, desc_df=desc_df, fp_df=fp_df, drop_cols=['SMILES', 'id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg'])
y_train = train_df[['SMILES', 'id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]


print(f"Merge all the Dataset \nX_train features : {X_train.shape[1]}\nY_train Features :{y_train.shape[1]}\n")


X_train

