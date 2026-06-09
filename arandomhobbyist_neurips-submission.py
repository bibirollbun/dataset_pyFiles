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


!pip install torch_molecule




import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

def assemble_dataframes():
    csv_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
    train_df = pd.read_csv(csv_path)

# 1. split off 20% for dev_test
    temp_df, dev_test = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,  # for reproducibility
        shuffle=True
    )

# 2. split the remaining 80% into 75% train / 25% valid â†’ 0.6 / 0.2 overall
    dev_train, dev_val = train_test_split(
        temp_df,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of the original
        random_state=42,
        shuffle=True
    )

    return dev_train, dev_val, train_df, temp_df, dev_test

dev_train, dev_val, train_df, temp_df, dev_test = assemble_dataframes()

class MaskedMAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        mask = ~torch.isnan(targets)
        masked_preds = preds[mask]
        masked_targets = targets[mask]
        return torch.mean(torch.abs(masked_preds - masked_targets))

print(f"Total rows:   {len(train_df)}")
print(f"Dev train:    {len(dev_train)} ({len(dev_train)/len(train_df):.2%})")
print(f"Dev valid:    {len(dev_val)} ({len(dev_val)/len(train_df):.2%})")
print(f"Dev test:     {len(dev_test)} ({len(dev_test)/len(train_df):.2%})")
print(f"Polymer example:{dev_train['SMILES'].to_list()[:3]}")
print(f"Columns:{dev_train.columns}")




import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
import torch  # if using PyTorch for your model

def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

def find_most_similar(test_fp, training_fps, training_smiles):
    # Calculate similarity of test_fp against all training_fps
    similarities = [TanimotoSimilarity(test_fp, train_fp) for train_fp in training_fps]
    # Find the index of the highest similarity
    max_idx = similarities.index(max(similarities))
    # Return best similarity and corresponding training SMILES
    return training_smiles[max_idx], similarities[max_idx]






import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm as notebook_tqdm
import tqdm
tqdm.tqdm = notebook_tqdm
tqdm.trange = notebook_tqdm
from torch_molecule import GNNMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_val_metrics(model, X, y_true):
    y_pred = np.array(model.predict(X))
    y_true = np.array(y_true)

    valid_mask = ~np.isnan(y_true)

    y_pred_valid = y_pred[valid_mask]
    y_true_valid = y_true[valid_mask]

    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    mse = mean_squared_error(y_true_valid, y_pred_valid)

    count = valid_mask.sum()                # total number of valid points

    return {
        "val_mae": mae,
        "mae":     mae,
        "mse":     mse,
        "r2":      r2,
        "mask":    int(count)}

def training(X_train, X_val, y_train, y_val, filepath):
    
    search_parameters = {
        'num_layer': ParameterSpec(
        param_type=ParameterType.INTEGER,
        value_range=(2, 5)
        ),
        'hidden_size': ParameterSpec(
        param_type=ParameterType.INTEGER,
        value_range=(64, 512)
        ),
        'learning_rate': ParameterSpec(
        param_type=ParameterType.LOG_FLOAT,
        value_range=(1e-4, 1e-2)
        ),
    }
    gnn = GNNMolecularPredictor(
    task_type="regression",
    num_task=5,
    loss_criterion=MaskedMAELoss(),  # ğŸ‘ˆ Plug your masked loss here
    batch_size=192,
    epochs=200,
    verbose=True
    )

    if os.path.exists(filepath):
        gnn.load_from_local(filepath)

    gnn.autofit(
    X_train = X_train,
    y_train = y_train,
    X_val = X_val,
    y_val = y_val,
    search_parameters=search_parameters,
    n_trials = 10 # number of times searching the best hyper-parameters
    )

    val_metrics = compute_val_metrics(model=gnn, X=X_val, y_true=y_val)

    gnn.save_to_local(filepath)
    
    return gnn

def create_submission(gnn, filepath):
    if os.path.exists(filepath):
        gnn.load_from_local(filepath)
        print("Model loaded successfully.")
    else:
        print("Model file not found.")

    test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

    X_test = test_df['SMILES'].to_list()
    ids = test_df['id'].to_list()  # <-- this was missing
    
    gnn_preds = gnn.predict(X_test)['prediction']

    if isinstance(gnn_preds, torch.Tensor):
        gnn_preds = gnn_preds.detach().cpu().numpy()

    if not isinstance(gnn_preds, np.ndarray):
        raise ValueError("â�Œ Predictions are not a NumPy array.")

    if gnn_preds.shape != (len(X_test), 5):
        raise ValueError(f"â�Œ Predictions shape {gnn_preds.shape} is invalid. Expected ({len(X_test)}, 5).")

    if np.isnan(gnn_preds).any():
        raise ValueError("â�Œ Predictions contain NaN values.")

    if np.isinf(gnn_preds).any():
        raise ValueError("â�Œ Predictions contain infinite values.")

    if np.allclose(gnn_preds, 0):
        raise ValueError("â�Œ Predictions are all zero â€” possibly a failed model.")

    # Optional: check realistic range
    if (gnn_preds < -1000).any() or (gnn_preds > 1e4).any():
        print("âš ï¸� Warning: Some predictions are outside realistic range.")

    submission_df = pd.DataFrame(gnn_preds, columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
    submission_df.insert(0, 'id', ids)

    submission_df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")


def meets_criterion(metrics, criterion, verbose=False):
    results = {key: (metrics.get(key, None), criterion[key]) for key in criterion}
    
    if verbose:
        print("\nğŸ”� Metric Check:")
        for key, (val, thresh) in results.items():
            if val is not None:
                print(f"  {key}: {val:.5f} (threshold: {thresh})")
            else:
                print(f"  {key}: MISSING")

    return all([
        results["mae"][0]     <= results["mae"][1],
        results["val_mae"][0] <= results["val_mae"][1],
        results["mask"][0]    >= results["mask"][1],
        results["mse"][0]     <= results["mse"][1],
        results["r2"][0]      >= results["r2"][1],
    ])

if __name__ == '__main__':
    criterion = {
    "val_mae": 0.065,
    "mae": 12,
    "mask": 1300,    # Assuming you want at least this many valid points
    "mse": 0.0093,
    "r2": 0.923
    }

    filepath = '/kaggle/working/gnn_polymer.pth'
    X_train = dev_train['SMILES'].to_list()
    y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
    X_val = dev_val['SMILES'].to_list()
    y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()

    best_score = -float('inf')
    best_model = None
    best_metrics = None
    filepath = '/kaggle/working/gnn_polymer.pth'  # adjust path as needed

    for i in range(10):
        val_metrics, gnn = training(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val, filepath=filepath)
        score = (
        val_metrics.get("r2", 0)
        - val_metrics.get("mae", 1)
        - val_metrics.get("mse", 1)
        )
        if score < best_score:
            score = best_score
            best_model = gnn
            best_metrics = val_metrics

    print("\nâœ… Final Best Model Evaluation:")
    if meets_criterion(best_metrics, criterion, verbose=True):
        create_submission(gnn=best_model, filepath=filepath)
        print("ğŸ“¤ Submission created successfully.")
    else:
        print("â�Œ No model met the performance criterion after 10 trials.")




    
    

