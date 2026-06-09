! python -m pip install --no-index --find-links=../input/packges-offline -r ../input/packges-offline/requirements.txt


import os
import optuna
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

### UNMODIFIABLE IMPORT BEGIN ###
import random
from pathlib import Path
from typing import Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# Define paths
source_dir = "/kaggle/input/neurips-open-polymer-prediction-2025"
base_dir = "/kaggle/working"
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

# Load files once
train_df = pd.read_csv(os.path.join(source_dir, "train.csv"))
test_df = pd.read_csv(os.path.join(source_dir, "test.csv"))
sample_df = pd.read_csv(os.path.join(source_dir, "sample_submission.csv"))

# Loop through each target
for target in targets:
    # Create output directory
    target_dir = os.path.join(base_dir, f"polymer_{target}", "competition",)
    os.makedirs(target_dir, exist_ok=True)

    # Copy test file as-is
    test_df.to_csv(os.path.join(target_dir, "test.csv"), index=False)

    # Prepare sample_submission with only current target
    sample_target_df = sample_df[["id", target]]
    sample_target_df.to_csv(os.path.join(target_dir, "sample_submission.csv"), index=False)

    # Prepare train file: only rows where target is not null, and only Id + target columns
    train_target_df = train_df[["id","SMILES", target]].dropna(subset=[target])
    train_target_df.to_csv(os.path.join(target_dir, "train.csv"), index=False)


test_df


sample_df


def smiles_to_features(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features = {
            "MolWt": Descriptors.MolWt(mol)
            ,"HeavyAtomMolWt": Descriptors.HeavyAtomMolWt(mol)
            ,"NumAtoms": mol.GetNumAtoms()
            ,"MolLogP": Descriptors.MolLogP(mol)
            ,"NumHDonors": Descriptors.NumHDonors(mol)
            ,"NumHAcceptors": Descriptors.NumHAcceptors(mol)
            ,"TPSA": Descriptors.TPSA(mol)
            ,"NumRotatableBonds": Descriptors.NumRotatableBonds(mol)
            ,"RingCount": Descriptors.RingCount(mol)
            ,"NumAromaticRings": Descriptors.NumAromaticRings(mol)
            ,"NumHeteroatoms": Descriptors.NumHeteroatoms(mol)
        }
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        fp = mfpgen.GetFingerprint(mol)
        features.update({f'FP_{i}': int(b) for i, b in enumerate(fp)})
        
        return features
    except:
        return None

def transform_data(target_columns, dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function to transform data into a format that can be used for training the model.
    Used on both Train and Test data. Test data may initially not contain target columns.
    """

    # Separating features and target if present
    data = dataset.copy(deep=True)

    features = data['SMILES'].apply(smiles_to_features)
    valid_smiles = features.notna()
    if not valid_smiles.any():
        raise ValueError("No valid SMILES strings found")
    data = data[valid_smiles].drop(columns=['SMILES'])
    features = features[valid_smiles].apply(pd.Series)
    data = pd.concat([data, features], axis=1)

    has_target = any(col in data.columns for col in target_columns)
    if has_target:
        features = data.drop(columns=target_columns)
        target = data[target_columns]
    else:
        features = data
        target = None

    return features, target


def optimize_booster(X_train, y_train, X_val, y_val, target_name):
    def objective(trial):
        model_type = trial.suggest_categorical("model", ["catboost", "lgbm", "xgb"])

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("lr", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        }

        if model_type == "catboost":
            model = CatBoostRegressor(**params, silent=True)
        elif model_type == "lgbm":
            model = LGBMRegressor(**params)
        else:
            model = XGBRegressor(**params)

        model.fit(X_train, y_train[target_name])
        return mean_absolute_error(y_val[target_name], model.predict(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=2)
    return study.best_params



# DATASET_PATH = Path("/kaggle/working/polymer_Density/competition")  # path for saving and loading dataset(s)

# TRAIN_FILE = DATASET_PATH / "train.csv"  # TODO: Replace with your actual filename
# TEST_FILE = DATASET_PATH / "test.csv"  # TODO: Replace with your actual filename

# train = pd.read_csv(TRAIN_FILE)
# X_test = pd.read_csv(TEST_FILE)
# train


# train_data, eval_test_data = train_test_split(
#     train, test_size=0.2, random_state=42
# )  


# target_columns = ["Density"]
# train_features, train_target = transform_data(train_data)
# eval_test_features, eval_test_target = transform_data(eval_test_data)
# test_features, _ = transform_data(X_test)


# eval_test_features


# print(f"\n=== Training {target} model ===")
# best_params = optimize_booster(train_features, train_target, eval_test_features, eval_test_target, target_columns[0])


# if best_params["model"] == "catboost":
#     model = CatBoostRegressor(
#         iterations=best_params["n_estimators"],
#         learning_rate=best_params["lr"],
#         depth=best_params["max_depth"],
#         subsample=best_params["subsample"],
#         # colsample_bylevel=best_params['colsample'],
#         loss_function="MAE",
#         verbose=False,
#     )
# elif best_params["model"] == "lgbm":
#     model = LGBMRegressor(
#         n_estimators=best_params["n_estimators"],
#         learning_rate=best_params["lr"],
#         max_depth=best_params["max_depth"],
#         subsample=best_params["subsample"],
#         # colsample_bytree=best_params['colsample'],
#         objective="mae",
#         random_state=42,
#     )
# else:
#     model = XGBRegressor(
#         n_estimators=best_params["n_estimators"],
#         learning_rate=best_params["lr"],
#         max_depth=best_params["max_depth"],
#         subsample=best_params["subsample"],
#         # colsample_bytree=best_params['colsample'],
#         eval_metric="mae",
#         random_state=42,
#     )

# model.fit(
#     train_features,
#     train_target[target_columns[0]],
# )


# preds = model.predict(eval_test_features)
# print("MAE", mean_absolute_error(eval_test_target[target_columns[0]], preds))
# print("R2", model.score(eval_test_features, eval_test_target[target_columns[0]]))


#preds


# data = {
#     'id': test_df['id'],
#     'A': np.array([1, 2, 3]),
#     'B': np.array([4, 5, 6]),
#     'C': np.array([7, 8, 9])
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# print(df)



targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

outputs = {}

for target in targets:
        
    DATASET_PATH = Path(f"/kaggle/working/polymer_{target}/competition")  # path for saving and loading dataset(s)
    
    TRAIN_FILE = DATASET_PATH / "train.csv"  # TODO: Replace with your actual filename
    TEST_FILE = DATASET_PATH / "test.csv"  # TODO: Replace with your actual filename
    
    train = pd.read_csv(TRAIN_FILE)
    X_test = pd.read_csv(TEST_FILE)
    train_data, eval_test_data = train_test_split(
        train, test_size=0.2, random_state=42
    )  # corresponding to 80%, 20% of ‘dataset‘

    target_cols = [target]
    train_features, train_target = transform_data(target_cols, train_data)
    eval_test_features, eval_test_target = transform_data(target_cols, eval_test_data)
    test_features, _ = transform_data(target_cols, X_test)

    print(f"\n=== Training {target} model ===")
    best_params = optimize_booster(train_features, train_target, eval_test_features, eval_test_target, target)
    
    if best_params["model"] == "catboost":
        model = CatBoostRegressor(
            iterations=best_params["n_estimators"],
            learning_rate=best_params["lr"],
            depth=best_params["max_depth"],
            subsample=best_params["subsample"],
            loss_function="MAE",
            verbose=False,
        )
    elif best_params["model"] == "lgbm":
        model = LGBMRegressor(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["lr"],
            max_depth=best_params["max_depth"],
            subsample=best_params["subsample"],
            objective="mae",
            random_state=42,
        )
    else:
        model = XGBRegressor(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["lr"],
            max_depth=best_params["max_depth"],
            subsample=best_params["subsample"],
            eval_metric="mae",
            random_state=42,
        )
    
    model.fit(
        train_features,
        train_target[target],
    )
    
    preds = model.predict(eval_test_features)
    print("MAE", mean_absolute_error(eval_test_target[target], preds))
    print("R2", model.score(eval_test_features, eval_test_target[target]))

    outputs[target] = model.predict(test_features)


submission = {'id': test_df['id']} | outputs
# Create DataFrame
submission = pd.DataFrame(submission)
submission


# Save final merged result
submission.to_csv(os.path.join(base_dir, '/kaggle/working/submission.csv'), index=False)

