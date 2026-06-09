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


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Load data
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# Quick EDA: target summary
print("Target Summary:")
print(train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].describe(), "\n")

# Histograms for each target
for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    plt.figure()
    train[col].hist(bins=30)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

# Feature engineering from SMILES
def featurize(smiles):
    features = {
        'len': len(smiles),
        'count_digits': sum(c.isdigit() for c in smiles),
        'count_paren_open': smiles.count('('),
        'count_paren_close': smiles.count(')'),
        'count_stars': smiles.count('*'),
    }
    for atom in ['C', 'O', 'N', 'F', 'S', 'P', 'H']:
        features[f'count_{atom}'] = smiles.count(atom)
    return features

X = pd.DataFrame([featurize(s) for s in train['SMILES']])
y = train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]


# Correlation matrix of features vs targets
corr = pd.concat([X, y], axis=1).corr()
corr_feat_targets = corr.loc[X.columns, y.columns]
print("Feature-Target Correlations:")
print(corr_feat_targets, "\n")

# Baseline model per-property Random Forest
print("Baseline MAE per property:")
for prop in y.columns:
    mask = y[prop].notnull()
    X_prop = X[mask]
    y_prop = y.loc[mask, prop]
    X_tr, X_val, y_tr, y_val = train_test_split(X_prop, y_prop, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"{prop}: {mae:.3f}")



# Load datasets
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

# Feature engineering function
def featurize(smiles):
    features = {
        'len': len(smiles),
        'count_digits': sum(c.isdigit() for c in smiles),
        'count_paren_open': smiles.count('('),
        'count_paren_close': smiles.count(')'),
        'count_stars': smiles.count('*'),
    }
    for atom in ['C', 'O', 'N', 'F', 'S', 'P', 'H']:
        features[f'count_{atom}'] = smiles.count(atom)
    return features

# Build feature matrices
X_train = pd.DataFrame([featurize(s) for s in train['SMILES']])
X_test  = pd.DataFrame([featurize(s) for s in test['SMILES']])

# Train a model per property and predict
properties = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
predictions = {}

for prop in properties:
    mask = train[prop].notnull()
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train[mask], train.loc[mask, prop])
    predictions[prop] = model.predict(X_test)

# Assemble submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Tg': predictions['Tg'],
    'FFV': predictions['FFV'],
    'Tc': predictions['Tc'],
    'Density': predictions['Density'],
    'Rg': predictions['Rg']
})

# Save to CSV
# submission.to_csv('/kaggle/working/submission.csv', index=False)

# Display the first few rows
print("Submission preview:")
print(submission.head())


import pandas as pd
import numpy as np
import re
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

# ——— Disable RDKit warnings ———
RDLogger.DisableLog('rdApp.*')

# ——— Fingerprint featurizer (strip '*', empty branches, stereo slashes, leading '=') ———
def mol_fp(smiles, radius=2, n_bits=1024):
    # remove asterisks, empty branches, stereo indicators, and stray leading '='
    clean = smiles.replace('*', '').replace('()', '')
    clean = re.sub(r'[\\/]', '', clean)      # remove / and \
    clean = re.sub(r'^=+', '', clean)        # strip leading '='
    
    try:
        mol = Chem.MolFromSmiles(clean, sanitize=False)
        if mol is None:
            raise ValueError("Could not parse")
        Chem.SanitizeMol(mol)  # explicit sanitization
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        # parsing or sanitization failed
        return np.zeros((n_bits,), dtype=int)


# ——— Build feature matrices ———
X_train_full = np.vstack(train['SMILES'].apply(mol_fp).values)
X_test       = np.vstack(test ['SMILES'].apply(mol_fp).values)

# ——— Train & predict per property ———
props = ['Tg','FFV','Tc','Density','Rg']
submission = pd.DataFrame({'id': test['id']})

for p in props:
    mask = train[p].notnull()
    Xp   = X_train_full[mask.values]
    yp   = train.loc[mask, p].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(Xp, yp)
    submission[p] = model.predict(X_test)

# ——— Save submission ———
submission.to_csv('/kaggle/working/submission.csv', index=False)
print(submission.head())

