


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')


train = train.fillna(0)  # Fill all missing values with 0
print("Missing values filled with 0.")
print("First 5 rows are as follows:")
print(train.head())



train.head()


train.tail()


TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
print(train[TARGETS].describe())

# ðŸ§ª Feature Extraction from SMILES
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([np.nan]*5)
    return pd.Series([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol)
    ])



feature_names = ['MolWt', 'MolLogP', 'RotBonds', 'TPSA', 'HDonors']
train[feature_names] = train['SMILES'].apply(extract_features)
test[feature_names] = test['SMILES'].apply(extract_features)


models = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = train[feature_names]
X_test = test[feature_names]

preds = np.zeros((len(test), len(TARGETS)))


for i, target in enumerate(TARGETS):
    y = train[target]
    fold_preds = np.zeros(len(test))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, y_train)
        dval = lgb.Dataset(X_val, y_val)

        model = lgb.train(
            {'objective': 'regression', 'metric': 'mae', 'verbosity': -1},
            dtrain,
            valid_sets=[dval],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )

        fold_preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
        models[f'{target}_fold{fold}'] = model

    preds[:, i] = fold_preds



submission = pd.DataFrame(preds, columns=TARGETS)
submission.insert(0, 'id', test['id'])
submission.to_csv('submission.csv', index=False)
submission.head()

