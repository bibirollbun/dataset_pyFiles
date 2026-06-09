!pip install rdkit torch_molecule


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_molecule import LSTMMolecularPredictor, GNNMolecularPredictor
from torch_molecule.utils.search import ParameterType, ParameterSpec
import tqdm
tqdm.tqdm = tqdm.notebook.tqdm
tqdm.trange = tqdm.notebook.tqdm


# Load and split data
train_df = pd.read_csv('//kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
temp_df, dev_test = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)
dev_train, dev_val = train_test_split(temp_df, test_size=0.25, random_state=42, shuffle=True)

X_train = header = dev_train['SMILES'].to_list()
y_train = dev_train[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
X_val = dev_val['SMILES'].to_list()
y_val = dev_val[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
X_test = dev_test['SMILES'].to_list()
y_test = dev_test[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()
task_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']



# Random Forest Model
def smiles_to_fp(smiles, radius=2, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits))

X_train_feats = np.vstack([smiles_to_fp(s) for s in X_train])
X_val_feats = np.vstack([smiles_to_fp(s) for s in X_val])
X_dev_feats = np.vstack([X_train_feats, X_val_feats])
y_dev = np.vstack([y_train, y_val])
X_test_feats = np.vstack([smiles_to_fp(s) for s in X_test])

rf_models = {}
rf_preds = np.zeros_like(y_test)
for idx, name in enumerate(task_names):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    mask = ~np.isnan(y_dev[:, idx])
    rf.fit(X_dev_feats[mask], y_dev[:, idx][mask])
    rf_models[name] = rf
    rf_preds[:, idx] = rf.predict(X_test_feats)


# Compute RF MSE
mse_rf = {}
for i, name in enumerate(task_names):
    mask = ~np.isnan(y_test[:, i])
    mse_rf[name] = mean_squared_error(y_test[mask, i], rf_preds[mask, i]) if mask.sum() > 0 else np.nan
mse_rf_overall = mean_squared_error(y_test[~np.isnan(y_test)], rf_preds[~np.isnan(y_test)])

print("RF MSE per task:", {name: f"{mse:.4f}" for name, mse in mse_rf.items()})
print(f"RF Overall MSE: {mse_rf_overall:.4f}")


# Submission using Random Forest only
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
X_test_submission = test_df['SMILES'].to_list()
X_test_submission_feats = np.vstack([smiles_to_fp(s) for s in X_test_submission])

rf_preds_sub = np.zeros((len(X_test_submission), len(task_names)))
for idx, name in enumerate(task_names):
    rf_preds_sub[:, idx] = rf_models[name].predict(X_test_submission_feats)

submission_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
submission_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']] = rf_preds_sub
submission_df.to_csv('submission.csv', index=False)




