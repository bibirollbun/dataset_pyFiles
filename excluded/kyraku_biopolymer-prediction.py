import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
sample = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")

targets = ['Density', 'Tc', 'Tg', 'Rg', 'FFV']


def smiles_to_features(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return np.array(fp)


X_test = np.array([smiles_to_features(smi) for smi in test["SMILES"]])


predictions = {}
mae_scores = {}


for target in targets:
    df = train[['SMILES', target]].dropna()
    n = len(df)
    X_train = np.array([smiles_to_features(smi) for smi in df['SMILES']])
    y_train = df[target].values

    model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    model.fit(X_train, y_train)

    
    y_pred_train = model.predict(X_train)
    predictions[target] = model.predict(X_test)

    
    mae = mean_absolute_error(y_train, y_pred_train)
    value_range = y_train.max() - y_train.min()

    mae_scores[target] = {"mae": mae, "range": value_range, "n": n}

    print(f"Trained {target} on {n} samples | MAE: {mae:.4f}")

# Compute wMAE
weights = {}
denominator = sum((1 / np.sqrt(score["n"])) / score["range"] for score in mae_scores.values())
for target, score in mae_scores.items():
    weight = ((1 / np.sqrt(score["n"])) / score["range"]) / denominator
    weights[target] = weight

wmae = sum(weights[t] * mae_scores[t]["mae"] for t in targets)
print(f"\nWeighted MAE (wMAE): {wmae:.4f}")


submission = sample.copy()
for target in targets:
    submission[target] = predictions[target]

submission.to_csv("submission.csv", index=False)
print("\nSubmission saved as: submission.csv")


