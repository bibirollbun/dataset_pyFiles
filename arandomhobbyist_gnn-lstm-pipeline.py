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

# 2. split the remaining 80% into 75% train / 25% valid → 0.6 / 0.2 overall
    dev_train, dev_val = train_test_split(
        temp_df,
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of the original
        random_state=42,
        shuffle=True
    )

    return dev_train, dev_val, train_df, temp_df, dev_test

dev_train, dev_val, train_df, temp_df, dev_test = assemble_dataframes()

print(f"Total rows:   {len(train_df)}")
print(f"Dev train:    {len(dev_train)} ({len(dev_train)/len(train_df):.2%})")
print(f"Dev valid:    {len(dev_val)} ({len(dev_val)/len(train_df):.2%})")
print(f"Dev test:     {len(dev_test)} ({len(dev_test)/len(train_df):.2%})")
print(f"Polymer example:{dev_train['SMILES'].to_list()[:3]}")
print(f"Columns:{dev_train.columns}")




!pip install --no-index --no-deps \
    /kaggle/input/kaggle-wheels/kaggle-wheels/numpy-2.3.2-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
    /kaggle/input/kaggle-wheels/kaggle-wheels/pillow-11.3.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl \
    /kaggle/input/kaggle-wheels/kaggle-wheels/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl \
    /kaggle/input/kaggle-wheels/kaggle-wheels/torch_geometric-2.6.1-py3-none-any.whl \
    /kaggle/input/kaggle-wheels/kaggle-wheels/torch_molecule-0.1.3-py3-none-any.whl


!ls /kaggle/input/gnn-model-files/
!pip install --no-index --find-links=/kaggle/input/gnn-model-files torch_molecule




import numpy, PIL, torch_geometric, torch_molecule
print("Numpy:", numpy.__version__)
print("Pillow:", PIL.__version__)

print("Torch Geometric:", torch_geometric.__version__)
print("Torch Molecule:", torch_molecule.__version__)


from torch_molecule import LSTMMolecularPredictor, GNNMolecularPredictor
from tqdm.notebook import tqdm as notebook_tqdm
import tqdm
tqdm.tqdm = notebook_tqdm
tqdm.trange = notebook_tqdm
from torch_molecule.utils.search import ParameterType, ParameterSpec
from scipy.optimize import minimize

def lstm_training(X_train, y_train, X_val, y_val):
    search_parameters = {
    "output_dim": ParameterSpec(ParameterType.INTEGER, (8, 32)),
    "LSTMunits": ParameterSpec(ParameterType.INTEGER, (30, 120)),
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-4, 1e-2)),
    }

    lstm = LSTMMolecularPredictor(
    task_type="regression",
    num_task=5,
    batch_size=192,
    epochs=200,
    verbose=True,
    )

    print("Model initialized successfully")

    lstm.autofit(
    X_train = X_train,
    y_train = y_train,
    X_val = X_val,
    y_val = y_val,
    search_parameters=search_parameters,
    n_trials = 10 # number of times searching the best hyper-parameters
    )

    return lstm

def gnn_training(X_train, y_train, X_val, y_val):
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
    batch_size=192,
    epochs=200,
    verbose=True,
    )

    print('Model initialized sucessfully')

    gnn.autofit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        search_parameters=search_parameters,
        n_trials=10
    )

    return gnn

def compute_wmae(y_true, y_pred):
    num_tasks = y_true.shape[1]
    #Range of each property
    ranges = np.nanmax(y_true, axis=0) - np.nanmin(y_true, axis=0)
    #Number of samples in a range...
    Ns = np.sum(~np.isnan(y_true), axis=0)
    #Inverse square weighting?
    raw_weights = 1.0/np.sqrt(Ns)
    #Normalize weights to 1?
    alphas = raw_weights/np.sum(raw_weights)

    wmae = 0.0

    for t in range(num_tasks):
        mask = ~np.isnan(y_true[:, t])  # only valid entries
        if np.sum(mask) == 0:
            continue
        mae_t = np.mean(np.abs(y_true[:, t] - y_pred[:, t]))/ranges[t]
        wmae += mae_t * alphas[t]

    return wmae


def find_optimal_weights(gnn_val, lstm_val, y_val, num_points=51):
    num_tasks = y_val.shape[1]

    def objective(weights, gnn_val, lstm_val, y_val):
        # Blend all tasks at once
        blended = weights * gnn_val + (1 - weights) * lstm_val
        return compute_wmae(y_val, blended)

    # Initial guess: 0.5 for all tasks
    x0 = np.full(num_tasks, 0.5)

    # Bounds for each weight: [0, 1]
    bounds = [(0, 1)] * num_tasks

    # Run optimizer
    res = minimize(objective, x0, args=(gnn_val, lstm_val, y_val), bounds=bounds)

    optimal_weights = res.x
    return optimal_weights

def create_submission(gnn, lstm, y_val, X_val):
    if gnn is not None:
        sample_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
        print(sample_sub.head())

        test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
        X_test = test_df['SMILES'].to_list()

        # Validation predictions
        gnn_val = gnn.predict(X_val)['prediction']
        lstm_val = lstm.predict(X_val)['prediction']

        # Find optimal blending weights on validation set
        weights = find_optimal_weights(gnn_val=gnn_val, lstm_val=lstm_val, y_val=y_val)

        print("Optimal blending weights per task:", weights)

        # Test predictions
        gnn_preds = gnn.predict(X_test)['prediction']
        lstm_preds = lstm.predict(X_test)['prediction']

        # Blend using optimal weights (vectorized)
        preds = weights * gnn_preds + (1 - weights) * lstm_preds

        # Create submission
        submission_df = sample_sub.copy()
        submission_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']] = preds

        print('submission_df', submission_df)
        submission_df.to_csv('submission.csv', index=False)



if __name__ == '__main__':
    X_train = dev_train['SMILES'].to_list()
    y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
    X_val = dev_val['SMILES'].to_list()
    y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()

    gnn = gnn_training(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print('GNN HAS FINISHED HIS EDUCATION. COMMENCING THE EDUCATION OF SIR LSTM...')
    lstm = lstm_training(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print('LSTM HAD FINISHED HIS EDUCATION. COMMENCING AP TESTING WITH HIEMLER...')
    create_submission(gnn=gnn, lstm=lstm, y_val=y_val, X_val=X_val)

