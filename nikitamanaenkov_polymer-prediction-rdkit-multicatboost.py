!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
import numpy as np 
import pandas as pd 
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")


gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)

def smile_features(df, smiles_column='SMILES'):
    features = []

    rdkit_desc_list = [
        'MolWt', 'MolLogP', 'TPSA', 'MolMR', 'HeavyAtomCount',
        'NumValenceElectrons', 'NumRadicalElectrons', 'LabuteASA',
        'BertzCT', 'BalabanJ', 'HallKierAlpha',
        'Chi0', 'Chi1', 'Chi2n', 'Chi3n'
    ]

    for smile in df[smiles_column]:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            data = {f'fp_{i}': 0 for i in range(4096)}
            data.update({k: 0 for k in rdkit_desc_list})
            features.append(data)
            continue

        fp = gen.GetFingerprint(mol)
        bitvect = fp.ToBitString()

        data = {}
        for i, bit_char in enumerate(bitvect):
            data[f'fp_{i}'] = int(bit_char)

        data.update({
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'TPSA': rdMolDescriptors.CalcTPSA(mol),
            'NumHDonors': Lipinski.NumHDonors(mol),
            'NumHAcceptors': Lipinski.NumHAcceptors(mol),
            'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
            'RingCount': rdMolDescriptors.CalcNumRings(mol),
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'FractionCSP3': rdMolDescriptors.CalcFractionCSP3(mol),
            'HeavyAtomCount': rdMolDescriptors.CalcNumHeavyAtoms(mol),
            'NHOHCount': Lipinski.NHOHCount(mol),
            'NOCount': Lipinski.NOCount(mol),
            'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol),
            'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
            'NumHeteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'MolMR': Descriptors.MolMR(mol),
            'LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),
            'BertzCT': Descriptors.BertzCT(mol),
            'BalabanJ': Descriptors.BalabanJ(mol),
            'HallKierAlpha': Descriptors.HallKierAlpha(mol),
            'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
            'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),
            'Chi0': Descriptors.Chi0(mol),
            'Chi1': Descriptors.Chi1(mol),
            'Chi2n': Descriptors.Chi2n(mol),
            'Chi3n': Descriptors.Chi3n(mol),
        })

        features.append(data)

    features_df = pd.DataFrame(features)
    return pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)



train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
train = smile_features(train)
test = smile_features(test)



target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
feature_cols = [col for col in train.columns if col not in ['id', 'SMILES'] + target_cols]

X_train = train[feature_cols]
y_train = train[target_cols]
X_test = test[feature_cols]

catboost_params = {
    'iterations': 2000,
    'learning_rate': 0.005,
    'depth': 6,
    'loss_function': 'RMSE',
    'verbose': 100,
    'random_seed': 28
}

N_SPLITS = 5

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

models = {}
predictions = {}
cv_scores = {}

for target in target_cols:
    print(f"\nTraining for target: {target}")
    mask = ~train[target].isna()
    X_sub = X_train[mask].reset_index(drop=True)
    y_sub = y_train.loc[mask, target].reset_index(drop=True)

    fold_models = []
    fold_preds = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_sub)):
        print(f"  Fold {fold + 1}/{N_SPLITS}")
        X_tr, X_val = X_sub.loc[train_idx], X_sub.loc[val_idx]
        y_tr, y_val = y_sub.loc[train_idx], y_sub.loc[val_idx]

        model = CatBoostRegressor(**catboost_params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)

        val_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_pred, squared=False)
        print(f"    Fold RMSE: {rmse:.4f}")
        fold_scores.append(rmse)

        fold_models.append(model)
        fold_preds += model.predict(X_test) / N_SPLITS

    models[target] = fold_models
    predictions[target] = fold_preds
    cv_scores[target] = {
        'mean_rmse': np.mean(fold_scores),
        'std_rmse': np.std(fold_scores)
    }



y_pred = pd.DataFrame(predictions)
submission = pd.DataFrame({'id': test['id']})
submission = pd.concat([submission, y_pred], axis=1)
submission.to_csv('submission.csv', index=False)


submission

