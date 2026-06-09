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


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor
import optuna

import warnings
warnings.filterwarnings("ignore")



# === 1. Load all datasets ===
train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
sample_submission = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")
train.head()


#Feature Extraction - RDKit Fingerprints

def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol

def get_fingerprint(smiles, n_bits=2048):
    mol = mol_from_smiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp)

print("Extracting fingerprints for train...")
X_fp_train = np.array([get_fingerprint(s) for s in train['SMILES']])
print("Extracting fingerprints for test...")
X_fp_test = np.array([get_fingerprint(s) for s in test['SMILES']])



#Targets
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


# Weighted MAE function
def weighted_mae(y_true_df, y_pred_df):
    ranges = {
        'Tg': 620.2797376,
        'FFV': 0.55010467,
        'Tc': 0.4775,
        'Density': 1.092307675,
        'Rg': 24.944550505
    }
    counts = {}
    maes = []
    total_weight = 0
    weighted_sum = 0
    for col in target_cols:
        mask = ~y_true_df[col].isna()
        counts[col] = mask.sum()
        if counts[col] > 0:
            mae = mean_absolute_error(y_true_df.loc[mask, col], y_pred_df.loc[mask, col])
            weight = 1 / ranges[col]
            weighted_sum += mae * weight
            total_weight += weight
    return weighted_sum / total_weight, counts, ranges



# Optuna Objective Function
def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "loss_function": "MAE",
        "verbose": False,
        "random_seed": 42
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)




# Model Training & Prediction

preds_train = pd.DataFrame(index=train.index, columns=target_cols)
preds_test = pd.DataFrame(index=test.index, columns=target_cols)

for col in target_cols:
    mask = ~train[col].isna()
    if mask.sum() == 0:
        continue
    
    X = X_fp_train[mask]
    y = train.loc[mask, col]
    
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optuna tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_tr, y_tr, X_val, y_val), n_trials=15)
    
    best_params = study.best_params
    best_params.update({"loss_function": "MAE", "verbose": False, "random_seed": 42})
    
    model = CatBoostRegressor(**best_params)
    model.fit(X, y)
    
    preds_train[col] = model.predict(X_fp_train)
    preds_test[col] = model.predict(X_fp_test)



# Evaluation
wmae_score, counts_used, ranges_used = weighted_mae(train[target_cols], preds_train)
print("=== Local competition-style weighted MAE ===")
print("Local wMAE:", wmae_score)
print("Counts used:", counts_used)
print("Ranges used:", ranges_used)


#Load sample_submission
sample_submission = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")

#submission

sample_submission[["Tg","FFV","Tc","Density","Rg"]]=preds_test

#save the prediction
sample_submission.to_csv("/kaggle/working/submission.csv", index=False)
print("saved: submission.csv")
sample_submission.head()




