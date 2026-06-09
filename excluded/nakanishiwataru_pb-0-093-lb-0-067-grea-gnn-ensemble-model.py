#Test offline install torch-molecule
!pip install /kaggle/input/torch-molecule-whl/torch_molecule-0.1.3-py3-none-any.whl --no-index --find-links=file:///kaggle/input/torch-molecule-pkg


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
#!pip install torch-molecule


import numpy as np
import pandas as pd
from torch_molecule import GREAMolecularPredictor, GNNMolecularPredictor

import shutil
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import argparse
import torch
from torch_molecule.utils.search import ParameterType, ParameterSpec

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops


search_GNN = {
    # Model architecture parameters
    "gnn_type": ParameterSpec(ParameterType.CATEGORICAL, ["gin-virtual", "gcn-virtual", "gin", "gcn"]),
    "norm_layer": ParameterSpec(ParameterType.CATEGORICAL, ["batch_norm", "layer_norm"],),
    "graph_pooling": ParameterSpec(ParameterType.CATEGORICAL, ["mean", "sum", "max"]),
    "augmented_feature": ParameterSpec(ParameterType.CATEGORICAL, ["maccs,morgan","maccs", "morgan", None]),
    "num_layer": ParameterSpec(ParameterType.INTEGER, (2, 5)),
    "hidden_size": ParameterSpec(ParameterType.INTEGER, (128, 512)),
    "drop_ratio": ParameterSpec(ParameterType.FLOAT, (0.0, 0.5)),
    "learning_rate": ParameterSpec(ParameterType.LOG_FLOAT, (1e-5, 1e-3)),
    "weight_decay": ParameterSpec(ParameterType.LOG_FLOAT, (1e-7, 1e-3)),
}

search_GREA = {
    "gamma": ParameterSpec(ParameterType.FLOAT, (0.25, 0.75)),
    **search_GNN
}


N_trial = 60
N_epoch = 250
BATCH_SIZE = 512


def train_and_evaluate_models(train_data, model_type='both'):
    #property_columns = ["Tg", "Tc", "Density", "Rg", "FFV"]
    property_columns = ["Tg", "Tc", "Density", "Rg"]
    print(f"Property columns: {property_columns}")

    # Train GREA model
    if model_type in ['grea', 'both']:
        grea_val_mae = {}
        #grea_h_vecs = {}
        for target in property_columns:
            print(f"Training GREA model for {target} properties...")
            notnull_train_data, notnull_valid_data = train_test_split(train_data[train_data[target].notnull()], test_size=0.1, random_state=23)
            X_train = notnull_train_data['SMILES']
            y_train = notnull_train_data[target].values

            X_val = notnull_valid_data['SMILES']
            y_val = notnull_valid_data[target].values
    
            grea_model = GREAMolecularPredictor(
                num_task=1,
                task_type="regression",
                model_name="GREA_singletask",
                batch_size=BATCH_SIZE,
                epochs=N_epoch,
                evaluate_criterion='mae',
                evaluate_higher_better=False,
                verbose=False
            )
        
            # Fit the model with hyperparameter optimization
            grea_model.autofit(
                X_train=X_train.tolist(),
                y_train=y_train,
                X_val=X_val.tolist(),
                y_val=y_val,
                n_trials=N_trial,
                search_parameters=search_GREA
            )
            # Recalc the val loss
            grea_val_output = grea_model.predict(X_val.tolist())
            #grea_h_vec = grea_val_output["representation"]
            #grea_h_vecs[target] = grea_h_vec
            
            val_mae = mean_absolute_error(y_val, grea_val_output['prediction'].flatten())
            grea_val_mae[target] = val_mae
            print(f"Property: {target}  Val MAE: {val_mae}")
        
            # Save the model
            grea_model.save(f"/kaggle/working/{grea_model.device}_GREA_{target}.pt")


    if model_type in ['gnn', 'both']:
        gnn_val_mae = {}
        #gnn_h_vecs = {}
        for target in property_columns:
            print(f"Training GNN model for {target} properties...")
            notnull_train_data, notnull_valid_data = train_test_split(train_data[train_data[target].notnull()], test_size=0.1, random_state=23)
            X_train = notnull_train_data['SMILES']
            y_train = notnull_train_data[target].values

            X_val = notnull_valid_data['SMILES']
            y_val = notnull_valid_data[target].values
            
            gnn_model = GNNMolecularPredictor(
                num_task=1,
                task_type="regression",
                model_name="GNN_singletask",
                batch_size=BATCH_SIZE,
                epochs=N_epoch,
                evaluate_criterion='mae',
                evaluate_higher_better=False,
                verbose=False
            )

            gnn_model.autofit(
                X_train=X_train.tolist(),
                y_train=y_train,
                X_val=X_val.tolist(),
                y_val=y_val,
                n_trials=N_trial,
                search_parameters=search_GNN
            )


            # Recalc the val loss
            gnn_val_output = gnn_model.predict(X_val.tolist())
            #gnn_h_vec = gnn_val_output["representation"]
            #gnn_h_vecs[target] = gnn_h_vec
            
            val_mae = mean_absolute_error(y_val, gnn_val_output['prediction'].flatten())
            gnn_val_mae[target] = val_mae
            print(f"Property: {target}  Val MAE: {val_mae}")
            
            # Save the model
            gnn_model.save(f"/kaggle/working/{gnn_model.device}_GNN_{target}.pt")

    if model_type in ['grea', 'both']:
        print('=== Grea Model Validation MAE ===')
        for target, value in grea_val_mae.items():
            print(f'{target}: {value}')
    if model_type in ['gnn', 'both']:
        print('=== GNN Model Validation MAE ===')
        for target, value in gnn_val_mae.items():
            print(f'{target}: {value}')
    return None


def test_models(train_data, test_data, model_type='both'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    property_columns = ["Tg", "Tc", "Density", "Rg", "FFV"]
    print(f"Property columns: {property_columns}")

    
    grea_model_weights = {
        "Tg":"/kaggle/input/grea-gnn-test/cuda_0_GREA_Tg.pt",
        "FFV":"/kaggle/input/grea-best-weights/GREA_BEST_FFV.pt",
        "Tc":"/kaggle/input/grea-best-weights/GREA_BEST_Tc.pt",
        "Density":"/kaggle/input/grea-best-weights/GREA_BEST_Density.pt",
        "Rg":"/kaggle/input/grea-best-weights/GREA_BEST_Rg.pt"
    }

    gnn_model_weights = {
        "Tg":"/kaggle/input/grea-gnn-test/cuda_0_GNN_Tg.pt",
        "FFV":"/kaggle/input/gnn-best-weight/GNN_FFV_BEST.pt",
        "Tc":"/kaggle/input/gnn-best-weight/GNN_Tc_BEST.pt",
        "Density":"/kaggle/input/gnn-best-weight/GNN_Density_BEST.pt",
        "Rg":"/kaggle/input/gnn-best-weight/GNN_Rg_BEST.pt"
    }

    # Predict GREA model
    if model_type in ['grea', 'both']:
        grea_result = {}
        grea_val_mae = {}
        grea_h_vecs = {}
        for target in property_columns:
            print(f"Test GREA model for {target} properties...")
            notnull_train_data, notnull_valid_data = train_test_split(train_data[train_data[target].notnull()], test_size=0.1, random_state=23)
            X_train = notnull_train_data['SMILES']
            y_train = notnull_train_data[target].values

            X_val = notnull_valid_data['SMILES']
            y_val = notnull_valid_data[target].values
            X_test = test_data['SMILES']
    
            grea_model = GREAMolecularPredictor()

            # Load best weight
            grea_model.cpu_load_from_local(device, grea_model_weights[target])

            # Calc Val MAE
            grea_val_output = grea_model.predict(X_val.tolist())
            val_mae = mean_absolute_error(y_val, grea_val_output['prediction'].flatten())
            grea_val_mae[target] = val_mae

            # Calc Test Prediction
            grea_output = grea_model.predict(X_test.tolist())
            grea_h_vec = grea_output["representation"]
            grea_h_vecs[target] = grea_h_vec
            
            grea_result[target] = grea_output['prediction'].flatten()

        #Make submission file
        grea_sub_df = pd.DataFrame(index=test_data['id'])

        for target, arr in grea_result.items():
            grea_sub_df[target] = arr


            

    # Predict GNN model
    if model_type in ['gnn', 'both']:
        gnn_result = {}
        gnn_val_mae = {}
        gnn_h_vecs = {}
        for target in property_columns:
            print(f"Test GNN model for {target} properties...")
            notnull_train_data, notnull_valid_data = train_test_split(train_data[train_data[target].notnull()], test_size=0.1, random_state=23)
            X_train = notnull_train_data['SMILES']
            y_train = notnull_train_data[target].values

            X_val = notnull_valid_data['SMILES']
            y_val = notnull_valid_data[target].values
            X_test = test_data['SMILES']
            
            gnn_model = GNNMolecularPredictor()

            # Load best weight
            gnn_model. cpu_load_from_local(device, gnn_model_weights[target])

            # Calc Val MAE
            gnn_val_output = gnn_model.predict(X_val.tolist())
            val_mae = mean_absolute_error(y_val, gnn_val_output['prediction'].flatten())
            gnn_val_mae[target] = val_mae

            # Calc Test Prediction
            gnn_output = gnn_model.predict(X_test.tolist())
            gnn_h_vec = gnn_output["representation"]
            gnn_h_vecs[target] = gnn_h_vec
            
            gnn_result[target] = gnn_output['prediction'].flatten()

        # Make Submission
        gnn_sub_df = pd.DataFrame(index=test_data['id'])

        for target, arr in gnn_result.items():
            gnn_sub_df[target] = arr


    print('=== Training Complete ===')
    if model_type in ['grea', 'both']:
        print('=== Grea Model Validation MAE ===')
        for target, value in grea_val_mae.items():
            print(f'{target}: {value}')
    if model_type in ['gnn', 'both']:
        print('=== GNN Model Validation MAE ===')
        for target, value in gnn_val_mae.items():
            print(f'{target}: {value}')
    return grea_sub_df, gnn_sub_df, grea_h_vecs, gnn_h_vecs


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()

    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Remove smiles that have target value in train_df from cross_smiles
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)
            
    # Fill Nan Value with cross smiles of extra data
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]

    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES: {len(unique_smiles_extra)}')

    return df_train

train_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test_data = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

data_tc = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
data_tc = data_tc.rename(columns={'TC_mean': 'Tc'})
data_tc = data_tc.reindex(['SMILES', "Tc"], axis=1)

data_tg2 = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
data_tg2 = data_tg2.rename(columns={'Tg (C)': 'Tg'})

data_tg3 = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
data_tg3 = data_tg3.rename(columns={'Tg [K]': 'Tg'})
data_tg3['Tg'] = data_tg3['Tg'] - 273.15

#data_tg4 = pd.read_csv("/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv")

data_dnst = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
data_dnst = data_dnst.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]

data_ffv = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')

train_data['SMILES'] = train_data['SMILES'].apply(lambda s: make_smile_canonical(s))
test_data['SMILES'] = test_data['SMILES'].apply(lambda s: make_smile_canonical(s))
data_dnst['SMILES'] = data_dnst['SMILES'].apply(lambda s: make_smile_canonical(s))
data_dnst = data_dnst[(data_dnst['SMILES'].notnull())&(data_dnst['Density'].notnull())&(data_dnst['Density'] != 'nylon')]
data_dnst['Density'] = data_dnst['Density'].astype('float64')
data_dnst['Density'] -= 0.118 # ??


train_data = add_extra_data(train_data, data_tc, 'Tc')
train_data = add_extra_data(train_data, data_tg2, 'Tg')
train_data = add_extra_data(train_data, data_tg3, 'Tg')
#train_data = add_extra_data(train_data, data_tg4, 'Tg')
train_data = add_extra_data(train_data, data_ffv, 'FFV')
train_data = add_extra_data(train_data, data_dnst, 'Density')



mode = "Test"

if mode == "Train":
    train_and_evaluate_models(train_data, model_type='both')
else:
    grea_sub_df, gnn_sub_df, grea_h_vecs, gnn_h_vecs = test_models(train_data, test_data, model_type='both')


print(grea_sub_df)
print(gnn_sub_df)


gnn_sub_df = (grea_sub_df + gnn_sub_df)/2


gnn_sub_df.to_csv('submission.csv')

