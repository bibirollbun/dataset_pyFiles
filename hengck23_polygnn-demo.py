# pip install
try: 
    import rdkit 
except:
    !pip -q install rdkit 

try:  
    import torch_geometric
except: 
    !pip -q install torch-geometric
    !pip install torch-scatter
    
# this takes time: Building wheels for collected packages: torch-scatter ...
print('PIP OK!!!')


import sys
sys.path.append('/kaggle/input/hengck23-polygnn-demo/lib')
sys.path.append('/kaggle/input/hengck23-polygnn-demo')

import pandas as pd
import torch
import polygnn
import polygnn_trainer as pt


print('IMPORT OK!!!')


# copied and modified from repo /polygnn-main/more_examples/example_predict.py

def make_prediction(data, pretain_model_dir):
    """
    Return the mean and std. dev. of a model prediction.

    Args:
        data (pd.DataFrame): The input data for the prediction.
        dir_name (str): The name of the directory containing the model that
            you desire to get predictions from. (e.g., "thermal", "electronic", etc.)
    """
    #root_dir = f"../trained_models/{dir_name}"
    root_dir = pretain_model_dir
    bond_config = polygnn.featurize.BondConfig(True, True, True)
    atom_config = polygnn.featurize.AtomConfig(
        True,
        True,
        True,
        True,
        True,
        True,
        combo_hybrid=False,  # if True, SP2/SP3 are combined into one feature
        aromatic=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # specify GPU

    # Load scalers
    scaler_dict = pt.load2.load_scalers(root_dir)

    # Load selectors
    selectors = pt.load2.load_selectors(root_dir)

    # Load and evaluate ensemble.
    ensemble = pt.load.load_ensemble(
        root_dir,
        polygnn.models.polyGNN,
        device,
        {
            "node_size": atom_config.n_features,
            "edge_size": bond_config.n_features,
            "selector_dim": len(selectors),
        },
    )
    print('loaded', ensemble.submodel_dict[0])
    
    # Define a lambda function for smiles featurization.
    smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
        x,
        bond_config,
        atom_config,
        "monocycle",
    ) 

    # Perform inference
    y, y_mean_hat, y_std_hat, _selectors = pt.infer.eval_ensemble(
        model=ensemble,
        root_dir=root_dir,
        dataframe=data,
        smiles_featurizer=smiles_featurizer,
        device=device,
        ensemble_kwargs_dict={"monte_carlo": False},
    )
    return y_mean_hat, y_std_hat


import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning, message="please use GetValence\(get Explicit=False\)")


valid_file='/kaggle/input/hengck23-polygnn-demo/Tg_valid.fold0.csv'
valid_df = pd.read_csv(valid_file)
truth = valid_df['Tg'].values

def fix_smiles(s): #chnage to psimles, [*] notation
    # Use regex to avoid already-correct '[*]'
    return re.sub(r'(?<!\[)\*(?!\])', '[*]', s)

valid_df['smiles_string']=valid_df['SMILES'].astype(str).apply(fix_smiles)
valid_df['prop']='exp_Tg__K'
pretain_model_dir= "/kaggle/input/hengck23-polygnn-demo/pretrained_models/thermal"
means, std_devs = make_prediction(valid_df[['smiles_string','prop']], pretain_model_dir)

#----

mae = np.abs( truth - means ).mean()
print('before fit', mae)

#----
#fit best median
def median_error_loss(params):
    m, b = params
    y_pred = m * means + b
    return np.mean(np.abs(truth - y_pred))


# model = LinearRegression().fit(means.reshape(-1, 1) , truth.reshape(-1, 1) )
# m = model.coef_[0]       # slope
# b = model.intercept_     # intercept

init = [0, 0]
result = minimize(median_error_loss, init, method='Nelder-Mead')
m, b = result.x



predict = m * means + b #-20
mae = np.abs(  predict - truth ).mean()
print('after fit:', mae)

plt.scatter(truth, predict)
plt.show()

