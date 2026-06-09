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
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')


# load the data
sum_path = "/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv"
train_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
test_path = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'

sample_submission_df = pd.read_csv(sum_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)


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


desc_df.sample(3)


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



def mol_to_morgan_fp(mol, radius=2, nBits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



train_df['mol'] = train_df['SMILES'].apply(Chem.MolFromSmiles)
fp_array = np.array([mol_to_morgan_fp(m) for m in train_df['mol']])
fp_df = pd.DataFrame(fp_array, columns=[f"FP_{i}" for i in range(fp_array.shape[1])])


fp_df.sample(3)


train_df = train_df.drop(columns=['mol'])


print(f"Descriptors Data Frame Shape :{desc_df.shape} \n")
print(f"Morgan Finger Print :{fp_df.shape} \n")
print(f"Train Data Frame :{train_df.shape} \n")


def combine_features(train_df, desc_df, fp_df, drop_cols=['SMILES']):
    combined = pd.concat([train_df.reset_index(drop=True), desc_df, fp_df], axis=1)
    combined = combined.drop(columns=drop_cols, errors='ignore')
    return combined


X_train = combine_features(train_df=train_df, desc_df=desc_df, fp_df=fp_df, drop_cols=['SMILES', 'id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg'])
y_train = train_df[['SMILES', 'id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]


print(f"Merge all the Dataset \nX_train features : {X_train.shape[1]}\nY_train Features :{y_train.shape[1]}\n")


def drop_high_nan_features(df, threshold=0.98):
    """
    Drop features with more than `threshold` proportion of NaN values.
    """
    nan_ratio = df.isna().mean()
    drop_cols = nan_ratio[nan_ratio > threshold].index.tolist()
    df_clean = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} features with >{threshold*100}% missing values.")
    return df_clean, drop_cols



from sklearn.feature_selection import VarianceThreshold

def drop_low_variance_features(df, threshold=0.0):
    """
    Remove features with variance below the threshold.
    Default 0.0 removes features with the same value everywhere.
    """
    selector = VarianceThreshold(threshold)
    selector.fit(df.fillna(0))  # Fill NaNs just for variance check

    kept_columns = df.columns[selector.get_support()]
    dropped_columns = [col for col in df.columns if col not in kept_columns]
    
    df_clean = df[kept_columns]
    print(f"Dropped {len(dropped_columns)} low-variance features.")
    return df_clean, dropped_columns



def drop_highly_correlated_features(df, threshold=0.95):
    """
    Remove one feature from each pair of features with correlation > threshold.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_clean = df.drop(columns=to_drop)

    print(f"Dropped {len(to_drop)} highly correlated features (>{threshold} correlation).")
    return df_clean, to_drop



# After combining features into X_train
print(f"Before Clean features : {X_train.shape[1]}")
X_train, nan_dropped = drop_high_nan_features(X_train, threshold=0.98)
X_train, var_dropped = drop_low_variance_features(X_train, threshold=0.0)
X_train, corr_dropped = drop_highly_correlated_features(X_train, threshold=0.95)
print(f"After Clean features : {X_train.shape[1]}")


X_test = combine_features(test_df, desc_df, fp_df, drop_cols=['SMILES', 'id'])


# After combining features into X_test
print(f"Before Clean features : {X_test.shape[1]}")
X_test, nan_dropped_t = drop_high_nan_features(X_test, threshold=0.98)
X_test, var_dropped_t = drop_low_variance_features(X_test, threshold=0.0)
X_test, corr_dropped_t = drop_highly_correlated_features(X_test, threshold=0.95)
print(f"After Clean features : {X_test.shape[1]}")


# Compute descriptors for test
descriptors_test = []
for smi in tqdm(test_df['SMILES'].to_list(), desc="Computing Descriptors for Test :"):
    descriptors_test.append(compute_all_descriptors(smi))

desc_test_df = pd.DataFrame(descriptors_test, columns=desc_names)

# Compute fingerprints for test
test_df['mol'] = test_df['SMILES'].apply(Chem.MolFromSmiles)
fp_array_test = np.array([mol_to_morgan_fp(m) for m in test_df['mol']])
fp_test_df = pd.DataFrame(fp_array_test, columns=[f"FP_{i}" for i in range(fp_array_test.shape[1])])

test_df = test_df.drop(columns=['mol'])
X_test = combine_features(test_df, desc_test_df, fp_test_df, drop_cols=['SMILES', 'id'])
X_test = X_test[X_train.columns]



targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
y = y_train[targets]

models = {}
predictions = pd.DataFrame()
predictions['id'] = test_df['id']

# For each target property
for target in tqdm(targets, desc="Working :"):
    # Keep only rows where this target is available
    idx = y[~y[target].isna()].index
    X_part = X_train.loc[idx].reset_index(drop=True)
    y_part = y.loc[idx, target].reset_index(drop=True)
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        tree_method='hist',
        random_state=42
    )
    model.fit(X_part, y_part)
    models[target] = model
    
    preds = model.predict(X_test)
    predictions[target] = preds

predictions.to_csv('submission.csv', index=False)


