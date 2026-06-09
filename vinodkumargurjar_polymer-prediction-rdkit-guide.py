!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


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


df_train=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
df_test=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
sample_submission=pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


df_train.head(5)


df_test.head(5)


from rdkit import Chem


df_train["SMILES"][0]


smiles = df_train["SMILES"][0]

# Convert to molecule object
mol = Chem.MolFromSmiles(smiles)
mol


# Reverse conversion: Mol -> SMILES
smiles_reversed = Chem.MolToSmiles(mol)
print(smiles_reversed)


from rdkit.Chem import Descriptors

print("Molecular Weight:", Descriptors.MolWt(mol))
print("LogP (hydrophobicity):", Descriptors.MolLogP(mol))
print("Number of Rings:", Descriptors.RingCount(mol))
print("Number of H-bond Acceptors:", Descriptors.NumHAcceptors(mol))
print("Number of Rotatable Bonds:", Descriptors.NumRotatableBonds(mol))


# Function to convert SMILES to a molecule object
def safe_mol(smiles):
    try:
        
        mol = Chem.MolFromSmiles(smiles)
        return mol  # Return the molecule
    except:
        return None  # If it fails, return None



# Apply to each SMILES in train and test datasets
df_train["mol"] = df_train["SMILES"].apply(safe_mol)
df_test["mol"] = df_test["SMILES"].apply(safe_mol)


df_train.head(2)


df_train['mol'][0]


df_test.head(2)


# # Drop any rows where the molecule couldn't be created (mol is None)
# df_train = df_train[df_train["mol"].notnull()].copy()
# df_test = df_test[df_test["mol"].notnull()].copy()


from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Create fingerprint generator globally (efficient for reuse)
generator = GetMorganGenerator(radius=2, fpSize=256)

def featurize_molecule(mol, n_bits=256):
    if mol is None:
        return None

    # Handcrafted descriptors
    features = {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumRings": Descriptors.RingCount(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
    }

    # Morgan fingerprint using new generator
    fp = generator.GetFingerprint(mol)
    bit_vector = list(fp.GetOnBits())
    
    for i in range(n_bits):
        features[f"FP_{i}"] = 1 if i in bit_vector else 0

    return pd.Series(features)



# Convert all molecules into numerical features
X_train = df_train["mol"].apply(featurize_molecule)
X_test = df_test["mol"].apply(featurize_molecule)


 # Keep ID so we know which row is which
X_train["id"] = df_train["id"].values
X_test["id"] = df_test["id"].values


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


X_train.head(5)


X_test.head(5)


X_train.columns.size,X_test.columns.size


target_columns= ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


df_train[target_columns].head(5)


df_train.tail(2)


df_train[target_columns].isnull().sum()


df_train[target_columns].notnull().sum()



# import optuna
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import mean_absolute_error

# # Define target columns
# targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

# # Objective function for Optuna
# def objective(trial, X, y):
#     params = {
#         "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
#         "max_depth": trial.suggest_int("max_depth", 5, 30),
#         "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
#         "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
#     }

#     model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
#     score = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=3)
#     return -np.mean(score)

# # Tune RandomForest for each target
# def tune_all_targets(X_train, df_train, targets, n_trials=10):
#     best_params_dict = {}

#     for target in targets:
#         print(f"\nğŸ”§ Tuning for target: {target}")
#         df_sub = df_train[~df_train[target].isna()]
#         X_sub = X_train[X_train["id"].isin(df_sub["id"])].drop(columns=["id"])
#         y_sub = df_sub[target]

#         # Skip if not enough samples
#         if len(y_sub) < 10:
#             print(f"âš ï¸� Skipping {target}: too few samples.")
#             continue

#         study = optuna.create_study(direction="minimize")
#         study.optimize(lambda trial: objective(trial, X_sub, y_sub), n_trials=n_trials)

#         print(f"âœ… Best params for {target}: {study.best_params}")
#         best_params_dict[target] = study.best_params

#     return best_params_dict

# # Run tuning
# best_params_per_target = tune_all_targets(X_train, df_train, targets, n_trials=30)



best_params_per_target={
    "Tg":{'n_estimators': 997, 'max_depth': 12, 
                'min_samples_split': 2, 'min_samples_leaf': 5},
"FFV":{'n_estimators': 488, 'max_depth': 22, 
                      'min_samples_split': 3, 'min_samples_leaf': 2},
"Tc":{'n_estimators': 759, 'max_depth': 16, 'min_samples_split': 3, 
                     'min_samples_leaf': 3},
"Density":{'n_estimators': 791, 'max_depth': 27, 
                          'min_samples_split': 3, 'min_samples_leaf': 2},

"Rg":{'n_estimators': 651, 'max_depth': 30, 'min_samples_split': 4, 
                     'min_samples_leaf': 3}}


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_and_predict_target(target_name, X_train, df_train, X_test):
    print(f"Training model for: {target_name}")
    
    # Filter rows where this target exists
    df_sub = df_train[~df_train[target_name].isna()]
    X_sub = X_train[X_train["id"].isin(df_sub["id"])]
    y_sub = df_sub[target_name]
    
    # Drop ID
    X_sub = X_sub.drop(columns=["id"])
    
    # Train
    model = RandomForestRegressor(**best_params_per_target[target_name],random_state=42)
    model.fit(X_sub, y_sub)    
    # Predict on test
    test_X = X_test.drop(columns=["id"])
    preds = model.predict(test_X)
    
    # Predict on train subset to get MAE
    train_preds = model.predict(X_sub)
    mae = mean_absolute_error(y_sub, train_preds)
    print(f"ğŸ§ª Train MAE ({target_name}): {mae:.5f}")
    
    return preds



targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
predictions = {}

for target in targets:
    preds = train_and_predict_target(target, X_train, df_train, X_test)
    predictions[target] = preds


submission = df_test[["id"]].copy()
for target in targets:
    submission[target] = predictions[target]


submission.head(5)


sample_submission.head()


submission.to_csv("submission.csv", index=False)




