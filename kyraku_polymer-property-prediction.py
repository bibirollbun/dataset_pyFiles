#train.csv

import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())


target_cols = ['Density', 'Tc', 'Tg', 'Rg', 'FFV']
print("\nMissing values in target columns:")
print(train[target_cols].isnull().sum())
print("\nSample rows:")
print(train.head())



print(train.shape)
print(train[['Density', 'Tc', 'Tg', 'Rg', 'FFV']].isnull().sum())



from rdkit import Chem

mol = Chem.MolFromSmiles("CCO")
print(mol is not None) 



import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
sample = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")


def smiles_to_features(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)


X_test = np.array([smiles_to_features(smi) for smi in test["SMILES"]])


targets = ['Density', 'Tc', 'Tg', 'Rg', 'FFV']
predictions = {}
mae_scores = {}


for target in targets:
    df = train[['SMILES', target]].dropna()
    X_train = np.array([smiles_to_features(smi) for smi in df['SMILES']])
    y_train = df[target].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    
    predictions[target] = model.predict(X_test)
    
    
    y_pred_train = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred_train)
    value_range = y_train.max() - y_train.min()
    n_samples = len(y_train)
    
    mae_scores[target] = {
        "mae": mae,
        "range": value_range,
        "n": n_samples
    }
    
    print(f" Trained {target} on {n_samples} samples | MAE: {mae:.4f}")


weights = {}
denominator = sum((1 / np.sqrt(score["n"])) / score["range"] for score in mae_scores.values())
for target, score in mae_scores.items():
    weight = ((1 / np.sqrt(score["n"])) / score["range"]) / denominator
    weights[target] = weight

wmae = sum(weights[t] * mae_scores[t]["mae"] for t in targets)
print(f"\n Weighted MAE (wMAE): {wmae:.4f}")


submission = sample.copy()
for target in targets:
    submission[target] = predictions[target]
submission.to_csv("submission.csv", index=False)
print("\n Submission saved as: submission.csv")



