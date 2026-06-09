!pip install /kaggle/input/rdkit-cp311/rdkit-2025.3.5-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install catboost


!pip install optuna


!pip install joblib


# ===================== Import Libraries =====================
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, DataStructs
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import optuna

from joblib import Parallel, delayed


# ===================== 1. Load Dataset =====================
train_path = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
test_path = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"

train_orig = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("\n===== TRAIN INFO =====")
print(f"Shape: {train_orig.shape}")
print(train_orig.dtypes)
print(train_orig.head())

print("\n===== TEST INFO =====")
print(f"Shape: {test.shape}")
print(test.dtypes)
print(test.head())

TARGET_COLS = [c for c in train_orig.columns if c not in ['id', 'SMILES']]


# ===================== 1b. Data Augmentation - Randomize SMILES =====================
from rdkit import Chem
import random
def randomize_smiles(smiles, random_seed=None):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        if random_seed is not None:
            random.seed(random_seed)
        return Chem.MolToSmiles(mol, doRandom=True)
    except Exception:
        return smiles

train_aug = train_orig.copy()
train_aug['SMILES'] = train_orig['SMILES'].apply(lambda x: randomize_smiles(x))
train = pd.concat([train_orig, train_aug], axis=0).reset_index(drop=True)


# ===================== 2. Molecule Feature Extractor Transformer (Parallelized) =====================
class MoleculeFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, radius=2, nBits=1024, n_jobs=-1):
        self.atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
        self.radius = radius
        self.nBits = nBits # Pastikan ini konsisten dengan penggunaan di bawah
        self.n_jobs = n_jobs

    def _get_features_from_smiles(self, smi):
        """Metode helper untuk mengekstrak fitur dari satu SMILES."""
        feats = {}
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # ... (Fitur yang sudah ada)
               
                # Tambahkan fitur baru
                feats['MolLogP'] = Descriptors.MolLogP(mol)
                feats['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
                feats['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
                feats['ExactMolWt'] = Descriptors.ExactMolWt(mol)
                feats['FractionCSP3'] = Descriptors.FractionCSP3(mol)
                feats['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
                feats['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
                feats['NumSaturatedHeterocycles'] = Descriptors.NumSaturatedHeterocycles(mol)
                feats['NumAromaticHeterocycles'] = Descriptors.NumAromaticHeterocycles(mol)
                feats['NumAmideBonds'] = rdMolDescriptors.CalcNumAmideBonds(mol)
                
                # === Gunakan self.nBits untuk konsistensi ===
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
                arr = np.zeros((self.nBits,), dtype=int)
                DataStructs.ConvertToNumpyArray(fp, arr)
                for i in range(self.nBits):
                    feats[f'fp_{i}'] = arr[i]
            else:
                raise ValueError("Invalid molecule")
        except Exception:
            # ... (Tambahkan fitur baru Anda di sini juga)
            feats = {f'fp_{i}': np.nan for i in range(self.nBits)}
            feats.update({f'count_{a}': np.nan for a in self.atoms})
            feats.update({'MolWt': np.nan, 'TPSA': np.nan, 'NumRotatableBonds': np.nan,
                          'NumHDonors': np.nan, 'NumHAcceptors': np.nan, 'NumHeavyAtoms': np.nan,
                          'NumAtoms': np.nan, 'NumRings': np.nan, 'count_ring': np.nan,
                          'length': np.nan})
            feats.update({'MolLogP': np.nan, 'NumAromaticRings': np.nan,
                          'NumHeteroatoms': np.nan, 'ExactMolWt': np.nan,
                          'FractionCSP3': np.nan, 'NumAliphaticRings': np.nan,
                          'NumSaturatedRings': np.nan, 'NumSaturatedHeterocycles': np.nan,
                          'NumAromaticHeterocycles': np.nan, 'NumAmideBonds': np.nan})
        return feats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"Extracting features using {self.n_jobs} cores...")
        features_list = Parallel(n_jobs=self.n_jobs)(
            delayed(self._get_features_from_smiles)(smi) for smi in X
        )
        return pd.DataFrame(features_list)



# ===================== 3. Prepare Features =====================
# Gunakan nBits yang lebih besar seperti yang disarankan sebelumnya
feat_extractor = MoleculeFeatureExtractor(radius=2, nBits=4096)

print("Extracting features from train SMILES...")
X_train_feats = feat_extractor.fit_transform(train['SMILES'])

print("Extracting features from test SMILES...")
X_test_feats = feat_extractor.transform(test['SMILES'])

# Hapus baris ini! Biarkan pipeline yang menangani pengisian NaN.
# X_train_feats.fillna(0, inplace=True)
# X_test_feats.fillna(0, inplace=True)

# Baris ini sudah benar, pastikan X_test_feats memiliki kolom yang sama.
X_test_feats = X_test_feats[X_train_feats.columns]

y = train[TARGET_COLS]


# ===================== 4. ML Pipeline Creation =====================
def make_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# Models for ensemble
def get_lgb_model():
    return lgb.LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

def get_xgb_model():
    return xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        tree_method="hist"
    )

def get_cat_model():
    return CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False
    )



# ===================== 4b. Hyperparameter Tuning dengan Optuna =====================

def tune_lgb_model(X, y):
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 2, 128),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }

        # Lakukan validasi silang mini untuk mendapatkan skor yang stabil
        model = lgb.LGBMRegressor(**params)
        kf_tune = KFold(n_splits=3, shuffle=True, random_state=42)
        mae_list = []
        for tr_idx, val_idx in kf_tune.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            pipe = make_pipeline(model)
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_val)
            mae_list.append(mean_absolute_error(y_val, pred))

        return np.mean(mae_list)

    print("Starting LGBM hyperparameter tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50) # Anda bisa mengubah jumlah trials

    print("Best LGBM parameters:", study.best_params)
    return study.best_params


# ===================== 5. Cross Validation Training =====================
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
n_targets = len(TARGET_COLS)

oof_preds_lgb = np.full((len(train), n_targets), np.nan)
oof_preds_xgb = np.full((len(train), n_targets), np.nan)
oof_preds_cat = np.full((len(train), n_targets), np.nan)

test_preds_lgb = np.zeros((len(test), n_targets))
test_preds_xgb = np.zeros((len(test), n_targets))
test_preds_cat = np.zeros((len(test), n_targets))

print("\nStarting CV training with ensemble pipelines...")

for ti, target in enumerate(TARGET_COLS):
    print(f"\nTarget [{ti+1}/{n_targets}]: {target}")

    mask_target = train[target].notnull()
    X = X_train_feats.loc[mask_target].values
    y_target = train.loc[mask_target, target].values

    oof_pred_lgb = np.zeros(len(y_target))
    oof_pred_xgb = np.zeros(len(y_target))
    oof_pred_cat = np.zeros(len(y_target))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{NFOLDS}")

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_target[tr_idx], y_target[val_idx]

        # LightGBM pipeline
        pipe_lgb = make_pipeline(get_lgb_model())
        pipe_lgb.fit(
            X_tr, y_tr,
            model__eval_set=[(X_val, y_val)],
            model__eval_metric="mae",
            model__callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        pred_val_lgb = pipe_lgb.predict(X_val)
        oof_pred_lgb[val_idx] = pred_val_lgb
        test_preds_lgb[:, ti] += pipe_lgb.predict(X_test_feats) / NFOLDS
        print(f"    - Fold {fold} MAE LGB: {mean_absolute_error(y_val, pred_val_lgb):.6f}")

        # XGBoost pipeline
        pipe_xgb = make_pipeline(get_xgb_model())
        pipe_xgb.fit(
            X_tr, y_tr,
            model__eval_set=[(X_val, y_val)],
            model__early_stopping_rounds=50,
            model__verbose=False
        )
        pred_val_xgb = pipe_xgb.predict(X_val)
        oof_pred_xgb[val_idx] = pred_val_xgb
        test_preds_xgb[:, ti] += pipe_xgb.predict(X_test_feats) / NFOLDS
        print(f"    - Fold {fold} MAE XGB: {mean_absolute_error(y_val, pred_val_xgb):.6f}")

        # CatBoost pipeline
        pipe_cat = make_pipeline(get_cat_model())
        pipe_cat.fit(
            X_tr, y_tr,
            model__eval_set=(X_val, y_val),
            model__early_stopping_rounds=50
        )
        pred_val_cat = pipe_cat.predict(X_val)
        oof_pred_cat[val_idx] = pred_val_cat
        test_preds_cat[:, ti] += pipe_cat.predict(X_test_feats) / NFOLDS
        print(f"    - Fold {fold} MAE CatBoost: {mean_absolute_error(y_val, pred_val_cat):.6f}")

    oof_preds_lgb[mask_target.values, ti] = oof_pred_lgb
    oof_preds_xgb[mask_target.values, ti] = oof_pred_xgb
    oof_preds_cat[mask_target.values, ti] = oof_pred_cat



# ===================== 6. Stacking (meta-model) =====================
print("\nTraining stacking meta-models...")

stacked_train = np.hstack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat])
stacked_test = np.hstack([test_preds_lgb, test_preds_xgb, test_preds_cat])

final_test_preds = np.zeros((len(test), n_targets))
stacking_models = {}
MIN_SAMPLES = 10

for i, target in enumerate(TARGET_COLS):
    mask_target = ~np.isnan(train[target].values)
    y_stack = train.loc[mask_target, target].values

    cols = [i, i + n_targets, i + 2 * n_targets]
    X_stack = stacked_train[mask_target][:, cols]

    valid_rows = ~np.isnan(X_stack).any(axis=1)
    X_stack_clean = X_stack[valid_rows]
    y_stack_clean = y_stack[valid_rows]

    if len(y_stack_clean) < MIN_SAMPLES:
        print(f"Warning: Only {len(y_stack_clean)} samples for stacking target '{target}', using base average.")
        final_test_preds[:, i] = stacked_test[:, cols].mean(axis=1)
        continue

    print(f"Training stacking model for '{target}' with {len(y_stack_clean)} samples.")
    meta_model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=200,      # Jumlah estimator yang cukup untuk meta-model
        learning_rate=0.05,
        num_leaves=16,
        max_depth=4,           # Kedalaman kecil karena fitur hanya 3
        random_state=42
    )

    meta_model.fit(X_stack_clean, y_stack_clean)

    pred_train_stack = meta_model.predict(X_stack_clean)
    train_mae = mean_absolute_error(y_stack_clean, pred_train_stack)
    print(f"  Train MAE stacking for '{target}': {train_mae:.6f}")

    X_test_stack = stacked_test[:, cols]
    if np.isnan(X_test_stack).any():
        X_test_stack = np.nan_to_num(X_test_stack)
    final_test_preds[:, i] = meta_model.predict(X_test_stack)
    stacking_models[target] = meta_model



# ===================== 7. Evaluate =====================
def weighted_mae(y_true, y_pred):
    weights = np.arange(1, y_true.shape[1] + 1)
    mae_per_target = []
    for i in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
        if np.sum(mask) == 0:
            mae = 0.0
        else:
            mae = np.mean(np.abs(y_true[mask, i] - y_pred[mask, i]))
        mae_per_target.append(mae)
    mae_per_target = np.array(mae_per_target)
    return np.sum(mae_per_target * weights) / np.sum(weights)

y_true_all = train[TARGET_COLS].values
oof_preds_ensemble = np.nanmean(np.stack([oof_preds_lgb, oof_preds_xgb, oof_preds_cat]), axis=0)

print(f"\nOOF Weighted MAE (Base Ensemble): {weighted_mae(y_true_all, oof_preds_ensemble):.6f}")

print(f"Final prediction shape: {final_test_preds.shape}")


# ===================== 8. Save Submission =====================
submission = pd.DataFrame(final_test_preds, columns=TARGET_COLS)
submission.insert(0, "id", test["id"].values)
submission.to_csv("submission.csv", index=False)
print("\nSubmission saved as submission.csv")
print(submission.head())




