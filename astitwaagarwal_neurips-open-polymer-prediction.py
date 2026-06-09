!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import pandas as pd
import numpy as np
import optuna
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, AllChem, Fragments
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


def generate_polymer_features(df, smiles_column='canonical_smiles', radius=2, n_bits=1024):
    def _calculate_features(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                'RotatableBonds': Descriptors.NumRotatableBonds(mol),
                'RingCount': rdMolDescriptors.CalcNumRings(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'EtherCount': Fragments.fr_ether(mol),
                'EsterCount': Fragments.fr_ester(mol),
                'AmideCount': Fragments.fr_amide(mol),
                'AromaticRingCount': rdMolDescriptors.CalcNumAromaticRings(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'BalabanJ': Descriptors.BalabanJ(mol),
            }
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            fp_features = {f'morgan_{i}': int(bit) for i, bit in enumerate(fp)}
            return {**descriptors, **fp_features}
        except Exception:
            return None

    features = []
    valid_indices = []
    for idx, smi in tqdm(enumerate(df[smiles_column]), total=len(df)):
        feat = _calculate_features(smi)
        if feat is not None:
            features.append(feat)
            valid_indices.append(idx)

    features_df = pd.DataFrame(features)
    result_df = pd.concat([df.iloc[valid_indices].reset_index(drop=True),
                           features_df.reset_index(drop=True)], axis=1)
    return result_df


print("ğŸ�¯ Loading data...")

train_path = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
test_path = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"
extra_SMILEStc_path = "/kaggle/input/tc-smiles/Tc_SMILES.csv"
extra_SMILEStg_path = "/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv"
extra_path = "/kaggle/input/smile-data/SMILES_EXTRA_DATA (1).csv"

# Load base data
test = pd.read_csv(test_path)
train = pd.read_csv(train_path, index_col="id")

# Load extra target-specific datasets
extra_tg = pd.read_csv(extra_SMILEStg_path, usecols=["SMILES", "Tg"])
extra_tc = pd.read_csv(extra_SMILEStc_path, usecols=["SMILES", "TC_mean"]).rename(columns={"TC_mean": "Tc"})

# Load additional SMILES data (already formatted like train)
extra = pd.read_csv(extra_path)

# Assign new IDs and index
next_id = train.index.max() + 1
extra = extra.set_index(pd.RangeIndex(next_id, next_id + len(extra), name="id"))

# Reorder columns to match train structure
extra = extra[train.columns]

# Combine into one dataset
overall_train = pd.concat([train, extra])

# Canonical SMILES function (define before this if not using Config class)
from rdkit import Chem

def get_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return ""
    except:
        return ""

# Apply canonical SMILES
print("\nGenerating canonical SMILES...")
overall_train['canonical_smiles'] = overall_train['SMILES'].fillna('').apply(get_canonical_smiles)
test['canonical_smiles'] = test['SMILES'].fillna('').apply(get_canonical_smiles)

# Summary
print(f"Original rows: {len(train)}")
print(f"New rows added: {len(extra)}")
print(f"Total rows: {len(overall_train)}")


train['canonical_smiles'] = train['SMILES'].fillna('')
test['canonical_smiles'] = test['SMILES'].fillna('')


train = generate_polymer_features(train)
test = generate_polymer_features(test)


TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
weights = {'Tg': 0.3, 'FFV': 0.175, 'Tc': 0.175, 'Density': 0.2, 'Rg': 0.15}


feature_columns = [col for col in train.columns if col not in TARGETS + ['SMILES', 'canonical_smiles', 'id']]

X = train[feature_columns].values
X_test = test[feature_columns].values


def objective(trial, X_train, y_train):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 300, 1000),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.4, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        'tree_method': 'hist',
        'random_state': 42
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=25, verbose=False)
        preds = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, preds))

    return np.mean(scores)


oof = np.zeros((len(train), len(TARGETS)))
preds = np.zeros((len(test), len(TARGETS)))
best_params_all = {}


for i, target in enumerate(TARGETS):
    print(f"\nğŸ”� Optimizing for target: {target}")
    y = train[target].values
    mask = ~np.isnan(y)
    X_valid, y_valid = X[mask], y[mask]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_valid, y_valid), n_trials=30, show_progress_bar=True)
    best_params = study.best_params
    best_params['tree_method'] = 'hist'
    best_params['random_state'] = 42

    print(f"âœ… Best params for {target}: {best_params}")
    best_params_all[target] = best_params

    # Re-train and generate predictions
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_valid)):
        model = XGBRegressor(**best_params)
        model.fit(X_valid[train_idx], y_valid[train_idx],
                  eval_set=[(X_valid[val_idx], y_valid[val_idx])],
                  early_stopping_rounds=25,
                  verbose=False)

        oof[val_idx, i] = model.predict(X_valid[val_idx])
        preds[:, i] += model.predict(X_test) / 5


mae_scores = {}
for i, target in enumerate(TARGETS):
    y_true = train[target].values
    mask = ~np.isnan(y_true)
    score = mean_absolute_error(y_true[mask], oof[mask, i])
    mae_scores[target] = score


weighted_mae = sum(mae_scores[t] * weights[t] for t in TARGETS)
print("\nğŸ“Š Individual MAEs:", mae_scores)
print(f"ğŸ�¯ Weighted MAE: {weighted_mae:.5f}")


submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
submission[targets] = preds
submission.to_csv('submission.csv', index=False)




