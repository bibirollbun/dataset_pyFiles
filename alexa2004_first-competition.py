# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

base_path = "/kaggle/input/neurips-open-polymer-prediction-2025"
for root, dirs, files in os.walk(base_path):
    for file in files:
        print(os.path.join(root, file))


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem, rdMolDescriptors, rdFingerprintGenerator
import matplotlib.pyplot as plt

# ----------- Load Data ------------
base_path = "/kaggle/input/neurips-open-polymer-prediction-2025"
train = pd.read_csv(f"{base_path}/train.csv")
test = pd.read_csv(f"{base_path}/test.csv")
targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

# ----------- Filter Valid SMILES ------------
def is_valid_smiles(smi):
    return Chem.MolFromSmiles(smi) is not None

train = train[train['SMILES'].apply(is_valid_smiles)].reset_index(drop=True)
test = test[test['SMILES'].apply(is_valid_smiles)].reset_index(drop=True)

# ----------- Feature Engineering ------------

# TF-IDF SMILES features
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=800)
X_ngram = vectorizer.fit_transform(train['SMILES']).toarray()
X_ngram_test = vectorizer.transform(test['SMILES']).toarray()

# RDKit descriptors
def rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {k: np.nan for k in [
            'MolWt', 'TPSA', 'LogP', 'NumRotatableBonds', 'NumHDonors',
            'NumHAcceptors', 'RingCount', 'FractionCSP3', 'HeavyAtomCount',
            'MolMR', 'Chi0', 'Chi1', 'Chi2n', 'Chi3n', 'BalabanJ'
        ]}
    return {
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LogP': Crippen.MolLogP(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'RingCount': Descriptors.RingCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'MolMR': Descriptors.MolMR(mol),
        'Chi0': Descriptors.Chi0n(mol),
        'Chi1': Descriptors.Chi1n(mol),
        'Chi2n': Descriptors.Chi2n(mol),
        'Chi3n': Descriptors.Chi3n(mol),
        'BalabanJ': Descriptors.BalabanJ(mol)
    }

df_rdkit = train['SMILES'].apply(rdkit_features).apply(pd.Series).fillna(0)
df_rdkit_test = test['SMILES'].apply(rdkit_features).apply(pd.Series).fillna(0)

# Morgan fingerprints (new API, no deprecation warning)
morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)

def morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(morgan_generator.GetNumBits(), dtype=int)
    fp = morgan_generator.GetFingerprint(mol)
    return np.array(list(fp.ToBitString()), dtype=int)


X_morgan = np.array([morgan_fp(s) for s in train['SMILES']])
X_morgan_test = np.array([morgan_fp(s) for s in test['SMILES']])

# Combine final features (TF-IDF + Morgan + RDKit)
X_all = np.hstack([X_ngram, X_morgan, df_rdkit.values])
X_all_test = np.hstack([X_ngram_test, X_morgan_test, df_rdkit_test.values])

feature_names = (
    [f"ngram_{i}" for i in range(X_ngram.shape[1])] +
    [f"morgan_{i}" for i in range(X_morgan.shape[1])] +
    list(df_rdkit.columns)
)

# ----------- Weighted MAE ------------
def compute_weights(train_df, targets):
    N, R, weights = {}, {}, {}
    for t in targets:
        values = train_df[t].dropna()
        N[t] = len(values)
        R[t] = values.max() - values.min()
    sum_weights = sum(1 / (np.sqrt(N[t]) * R[t]) for t in targets)
    for t in targets:
        weights[t] = (1 / (np.sqrt(N[t]) * R[t])) / sum_weights
    return weights, N, R

def weighted_mae(y_true_df, y_pred_df, weights, N, targets):
    total_error = 0
    for t in targets:
        mask = ~y_true_df[t].isna()
        mae = np.abs(y_true_df.loc[mask, t] - y_pred_df.loc[mask, t]).mean()
        total_error += weights[t] * mae
    return total_error

# ----------- Model Training ------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = pd.DataFrame(np.nan, index=train.index, columns=targets)
submission = pd.DataFrame({'id': test['id']})
all_importances = {}

for target in targets:
    print(f"\nðŸ“Š Training model for: {target}")
    mask = ~train[target].isna()
    X_train_all = X_all[mask]
    y_train_all = train.loc[mask, target].values
    idx_train_all = train.loc[mask].index

    test_preds = np.zeros(len(test))
    importances = np.zeros(X_all.shape[1])

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_all)):
        X_tr, X_val = X_train_all[train_idx], X_train_all[val_idx]
        y_tr, y_val = y_train_all[train_idx], y_train_all[val_idx]

        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        print(f"  Fold {fold+1} RMSE: {mean_squared_error(y_val, val_pred, squared=False):.4f}")

        oof_preds.loc[idx_train_all[val_idx], target] = val_pred
        test_preds += model.predict(X_all_test) / kf.n_splits
        importances += model.feature_importances_ / kf.n_splits

    submission[target] = test_preds
    all_importances[target] = importances

    # Optional: plot feature importances
    if target == "Tg":
        top_idx = np.argsort(importances)[-20:]
        plt.figure(figsize=(8, 6))
        plt.barh(range(20), importances[top_idx])
        plt.yticks(range(20), [feature_names[i] for i in top_idx])
        plt.xlabel("Avg Feature Importance")
        plt.title(f"Top Features for {target}")
        plt.tight_layout()
        plt.show()

# ----------- Evaluate & Save ------------
weights, N, R = compute_weights(train, targets)
wmae_score = weighted_mae(train, oof_preds, weights, N, targets)
print(f"\nâœ… Weighted MAE on OOF predictions: {wmae_score:.6f}")

submission.to_csv("submission.csv", index=False, float_format="%.4f")
print("âœ… submission.csv saved.")


test = pd.read_csv(f"/kaggle/working/submission.csv")
print(test)

