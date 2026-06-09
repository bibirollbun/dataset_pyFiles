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


# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
!pip install /kaggle/input/external-packages/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
print("Setup Complete")


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
print("Package Setup Complete")


train_df = pd.read_csv("../input/neurips-open-polymer-prediction-2025/train.csv")
test_df = pd.read_csv("../input/neurips-open-polymer-prediction-2025/test.csv")
sample_submission = pd.read_csv("../input/neurips-open-polymer-prediction-2025/sample_submission.csv")
print("Loading Data Set Complete")



#Target columns
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
#targets = 'FFV'

#train_df = train_df[train_df[targets].notnull()].copy()
#for col in targets:
#    train_df[col].fillna(train_df[col].mean(), inplace=True)

train_df.fillna({col: train_df[col].median() for col in targets}, inplace=True)

# SMILES featurization: Morgan fingerprints (radius=2, 2048 bits)
def smiles_to_morgan(smiles_list, radius=2, n_bits=2048, verbose=False):
    fingerprints = []
    valid_indices = []

    for idx, smi in enumerate(tqdm(smiles_list)):
        try:
            mol = Chem.MolFromSmiles(smi.replace("nan", "0"))
            if mol is None:
                if verbose:
                    print(f"âš ï¸� Invalid SMILES at index {idx}: {smi}")
                continue
            arr = np.zeros((n_bits,))
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
            valid_indices.append(idx)
        except Exception as e:
            if verbose:
                print(f"â�Œ Error at index {idx}: {smi} â€” {str(e)}")
            continue

    return np.array(fingerprints)

# Featurize train and test sets
X_train_fp = smiles_to_morgan(train_df["SMILES"], verbose=True)
X_test_fp = smiles_to_morgan(test_df["SMILES"], verbose=True)
#y_train_ffv = train_df[target]

# Train and predict
test_preds = {}
models = {}

for target in targets:
    y_train_ffv = train_df[target]

    # Train/Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_fp, y_train_ffv, test_size=0.2, random_state=42)

    # Train XGBoost
    model = xgb.XGBRegressor(early_stopping_rounds=25,
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Predict on test set
    ffv_preds = model.predict(X_test_fp)
    test_preds[target] = ffv_preds
    models[target] = model

print("Training and prediction for all targets completed")


# Prepare submission
submission_df = sample_submission.copy()
for target in targets:
    submission_df[target] = test_preds[target]
    
# Save output
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)

print(f"âœ… Submission saved to {output_path}")

