%%capture
# TabPFN
!pip install -q --no-index --find-links /kaggle/input/tabpfn-v2-0-9 tabpfn
!mkdir -p /root/.cache/tabpfn/
!cp /kaggle/input/tabpfn-v2-0-9/tabpfn-v2-regressor.ckpt /root/.cache/tabpfn/tabpfn-v2-regressor.ckpt


%%capture
# RDKit
!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import os
import torch

import numpy as np 
import pandas as pd 

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold

from tabpfn import TabPFNRegressor

import warnings
warnings.filterwarnings("ignore")


%%capture
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load datasets
train_df = pd.read_csv("../input/neurips-open-polymer-prediction-2025/train.csv")
test_df = pd.read_csv("../input/neurips-open-polymer-prediction-2025/test.csv")


# Print number rows in each dataset
print(f"Rows in training set: {train_df.shape[0]}")
print(f"Rows in test set: {test_df.shape[0]}")
print(f"Total rows of data: {train_df.shape[0] + test_df.shape[0]}")


# Print targets
targets = train_df.drop(columns=["id", "SMILES"]).columns.tolist()
print(f"Targets to predict: {targets}")


from rdkit.Chem import Descriptors

# Descriptor names and functions
desc_names, desc_funcs = zip(*Descriptors.descList)

# Compute descriptors for a SMILES
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [f(mol) if mol else None for f in desc_funcs]

# Compute descriptors for datasets
descriptors_train = [compute_descriptors(smi) for smi in train_df['SMILES']]
descriptors_test = [compute_descriptors(smi) for smi in test_df['SMILES']]


# extra_tg_df = pd.read_csv('/kaggle/input/smiles-tg/Tg_SMILES_class_pid_polyinfo_median.csv')
# display(extra_tg_df.head(3))

extra_tc_df = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
display(extra_tc_df.head(3))


import pandas as pd

# Assuming train_df and extra_tc_df are already loaded
# Example placeholders if you're testing this snippet independently:
# data = {'SMILES': ['C', 'CC'], 'Tg': [200, 250], 'FFV': [0.5, 0.6], 'Tc': [150, 180], 'Density': [1.0, 0.9], 'Rg': [10, 12], 'id': [0, 1]}
# train_df = pd.DataFrame(data)
#
# extra_tc_data = {'SMILES': ['CCC', 'CCCC'], 'TC_mean': [200, 230]}
# extra_tc_df = pd.DataFrame(extra_tc_data)


# Prepare extra_tc_df
extra_tc_clean = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})

# Generate 'id' for extra_tc_clean, starting directly after train_df's last id
start_id_for_tc = len(train_df)
extra_tc_clean['id'] = range(start_id_for_tc, start_id_for_tc + len(extra_tc_df))

extra_tc_clean[['Tg', 'FFV', 'Density', 'Rg']] = float('nan')

# Reorder columns to match train_df
extra_tc_clean = extra_tc_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Combine datasets into train_df
# Note: extra_tg_clean is no longer included
train_combined_df = pd.concat([train_df, extra_tc_clean], ignore_index=True)

# Assuming 'targets' is a list of your target columns, e.g., ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg'] # Define targets if not already defined

print(train_combined_df[targets].count())


# # Prepare extra_tg_df
# extra_tg_clean = extra_tg_df[['SMILES', 'PID', 'Tg']].rename(columns={'PID': 'id'})
# extra_tg_clean[['FFV', 'Tc', 'Density', 'Rg']] = float('nan')

# # Prepare extra_tc_df  
# extra_tc_clean = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})
# extra_tc_clean['id'] = range(len(train_df) + len(extra_tg_df), len(train_df) + len(extra_tg_df) + len(extra_tc_df))
# extra_tc_clean[['Tg', 'FFV', 'Density', 'Rg']] = float('nan')

# # Reorder columns to match train_df
# extra_tg_clean = extra_tg_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
# extra_tc_clean = extra_tc_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# # Combine all datasets into train_df
# train_combined_df = pd.concat([train_df, extra_tg_clean, extra_tc_clean], ignore_index=True)

# print(train_combined_df[targets].count())


# Prepare the features dataset
X = (
    pd.DataFrame(descriptors_train, columns=desc_names) # Add descriptors (features)
    .replace([np.inf, -np.inf], np.nan) # Replace inf values with nan
)

features = X.columns.tolist()
print(f"Total features: {len(features)}")


# Prepare the targets dataset
y = (train_df
     .replace([np.inf, -np.inf], np.nan) # Replace inf values with nan
     .drop(columns=["id", "SMILES"]) # Drop columns that aren't targets
)
print(f"Total targets: {y.shape[1]}")


# Prepare the test dataset
y_test = (
    pd.concat([test_df, pd.DataFrame(descriptors_test, columns=desc_names)], axis=1) # Get id, SMILES, and add descriptors
    .replace([np.inf, -np.inf], np.nan)
)


%%time

metrics = []
models = {}

def kfold_train_and_evaluate(X, y, target, n_splits=5, row_to_remove=None, device='cuda'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Remove row 408 from the Density dataset, as it contains corrupted data
        if row_to_remove:
            X_train, y_train = X_train.tail(row_to_remove), y_train.tail(row_to_remove)
            X_val, y_val = X_val.tail(row_to_remove), y_val.tail(row_to_remove)

        # Use GPU
        reg = TabPFNRegressor(n_estimators=25, device=device, random_state=42)
        # Fit
        reg.fit(X_train, y_train)

        # Predict (no_grad just in case)
        with torch.no_grad():
            y_pred = reg.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        fold_maes.append(mae)
        fold_models.append(reg)

        print(f"  Fold {fold+1}/{n_splits} MAE: {mae:.4f}")

    best_idx = np.argmin(fold_maes)
    best_model = fold_models[best_idx]
    best_mae = fold_maes[best_idx]
    avg_mae = np.mean(fold_maes)

    return best_model, best_mae, avg_mae, X_train.shape[0]

# Main loop per target
for target in targets:
    print(f"\nTraining {target}...")

    # Drop NaNs
    mask = y.dropna(subset=[target])
    idx = mask.index
    X_target = X.loc[idx]
    y_target = mask[[target]]

    row_to_remove = 408 if target == "FFV" else None

    best_model, best_mae, avg_mae, train_size = kfold_train_and_evaluate(
        X_target, y_target, target, n_splits=20, row_to_remove=row_to_remove, device='cuda'
    )

    metrics.append({
        "target": target,
        "best_mae": best_mae,
        "avg_mae": avg_mae,
        "train_size": train_size
    })

    models[target] = best_model

    print(f"{target} BEST MAE: {best_mae:.4f} | AVG MAE: {avg_mae:.4f}")


# Examine performance
pd.DataFrame(metrics)


# Examine our models object
models


# Start result DataFrame with IDs
y_preds = pd.DataFrame({'id': y_test.id})

# Loop through each model and generate predictions on test set
for target, model in models.items():
    y_preds[target] = model.predict(y_test[features])

# Examine predictions.
y_preds


# Submit predictions
y_preds.to_csv("submission.csv", index=False)

