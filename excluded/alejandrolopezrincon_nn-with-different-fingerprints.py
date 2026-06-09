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


!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator, GetTopologicalTorsionGenerator
from rdkit.Chem import MACCSkeys
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim



# Normalization parameters
normalization_params = {
    'Tg': {'mean': 96.452, 'std': 111.119},
    'FFV': {'mean': 0.367, 'std': 0.030},
    'Tc': {'mean': 0.256, 'std': 0.089},
    'Density': {'mean': 0.985, 'std': 0.146},
    'Rg': {'mean': 16.420, 'std': 4.605}
}

def normalize_targets(y, label):
    params = normalization_params[label]
    return (y - params['mean']) / params['std']

def denormalize_predictions(y_pred, label):
    params = normalization_params[label]
    return y_pred * params['std'] + params['mean']

def separate_subtables(file_path):
    df = pd.read_csv(file_path)
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    subtables = {}
    for label in labels:
        subtables[label] = df[['id', 'SMILES', label]][df[label].notna()]
    return subtables



def smiles_to_combined_fingerprints(smiles_list, radius=2, n_bits=64):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    atom_pair_gen = GetAtomPairGenerator(fpSize=n_bits)
    torsion_gen = GetTopologicalTorsionGenerator(fpSize=n_bits)

    fingerprints = []
    valid_smiles = []
    invalid_indices = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            morgan_fp = generator.GetFingerprint(mol)
            atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            torsion_fp = torsion_gen.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)

            combined_fp = np.concatenate([
                np.array(morgan_fp),
                np.array(atom_pair_fp),
                np.array(torsion_fp),
                np.array(maccs_fp)
            ])
            fingerprints.append(combined_fp)
            valid_smiles.append(smiles)
        else:
            # Total length = 3 * n_bits + 167 (MACCS)
            fingerprints.append(np.zeros(n_bits * 3 + 167))
            valid_smiles.append(None)
            invalid_indices.append(i)

    return np.array(fingerprints), valid_smiles, invalid_indices





def remove_constant_columns(X):
    constant_columns = np.all(X == X[0, :], axis=0)
    removed_indices = np.where(constant_columns)[0]
    X_filtered = X[:, ~constant_columns]
    return X_filtered, removed_indices

class FingerprintNN(nn.Module):
    def __init__(self, input_dim):
        super(FingerprintNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_and_predict(label, subtable, test_df, device):
    smiles_list = subtable['SMILES'].tolist()
    y = subtable[label].values
    X, _, invalids = smiles_to_combined_fingerprints(smiles_list)
    y = np.delete(y, invalids)

    if len(y) < 5:
        print(f"Skipping {label} due to insufficient data.")
        return None

    X, removed_indices = remove_constant_columns(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = normalize_targets(y, label)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        model = FingerprintNN(X_tensor.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()

        best_rmse = float('inf')
        no_improve_epochs = 0
        eval_counter = 0
        max_epochs = 10000
        early_stop_patience = 250
        rmse_threshold = 1e-4

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_tensor[train_idx])
            loss = loss_fn(output, y_tensor[train_idx])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                pred = model(X_tensor[test_idx]).cpu().numpy()
                true = y_tensor[test_idx].cpu().numpy()
                rmse = np.sqrt(mean_squared_error(true, pred))
                eval_counter += 1

            if rmse < best_rmse:
                best_rmse = rmse
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if best_rmse < rmse_threshold or no_improve_epochs >= early_stop_patience:
                break

        results.append(best_rmse)
        print(f"{label} - Fold {fold+1} RMSE: {best_rmse:.4f} ({eval_counter} evals)")

    print(f"{label} - Average RMSE: {np.mean(results):.4f}")

    # Predict on test set
    test_smiles = test_df['SMILES'].str.replace('*', 'C')
    test_ids = test_df['id'].values
    X_test, _, _ = smiles_to_combined_fingerprints(test_smiles)
    X_test = np.delete(X_test, removed_indices, axis=1)
    X_test = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy().flatten()
        predictions = denormalize_predictions(predictions, label)

    return pd.DataFrame({'id': test_ids, label: predictions})


 print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_file = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
test_file = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
subtables = separate_subtables(train_file)
test_df = pd.read_csv(test_file)
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
results = []

for label in labels:
    print(f"Processing label: {label}")
    result = train_and_predict(label, subtables[label], test_df, device)
    if result is not None:
        results.append(result)

if results:
    merged = results[0]
    for df in results[1:]:
        merged = pd.merge(merged, df, on='id')
    merged.to_csv("submission.csv", index=False)
    print("Saved predictions to submission.csv")


print(merged)

