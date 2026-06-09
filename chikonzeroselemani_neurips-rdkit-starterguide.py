!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import optuna
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Fragments, AllChem
from pathlib import Path
import logging
import warnings

# Completely silence all warnings and logging
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)

# ====================
# Configuration
# ====================
class Config:
    RANDOM_STATE = 42
    USE_GPU = True
    GPU_ID = 0
    N_FOLDS = 5
    MAX_ITERATIONS = 10000
    EARLY_STOPPING = 100
    OPTUNA_TRIALS = 100
    VERBOSE_EVAL = 100
    TARGET_COLUMNS = ["Tg", "FFV", "Tc", "Density", "Rg"]
    STUDY_DIR = Path("studies")
    MODEL_DIR = Path("models")
    
    def __init__(self):
        self.STUDY_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def get_canonical_smiles(smiles):
            """Convert SMILES to canonical form for consistent grouping"""
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
                return smiles  # fallback to original if conversion fails
            except:
                return smiles


print("ğŸ�¯ Loading data...")

train_path = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
test_path = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"
extra_SMILEStc_path = "/kaggle/input/tc-smiles/Tc_SMILES.csv"
extra_SMILEStg_path = "/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv"

# Load data
test = pd.read_csv(test_path)
train = pd.read_csv(train_path, index_col="id")
extra_tg = pd.read_csv(extra_SMILEStg_path, usecols=["SMILES", "Tg"])
extra_tc = pd.read_csv(extra_SMILEStc_path, usecols=["SMILES", "TC_mean"]).rename(columns={"TC_mean": "Tc"})

train.head(5)


# #  Filter out existing SMILES
# existing_smiles = set(train["SMILES"])
# extra_tg = extra_tg[~extra_tg["SMILES"].isin(existing_smiles)].copy()
# extra_tc = extra_tc[~extra_tc["SMILES"].isin(existing_smiles)].copy()

# #  Merge Tg and Tc data (outer merge to keep all SMILES)
# extra = pd.merge(
#     extra_tg,
#     extra_tc,
#     on="SMILES",
#     how="outer",
# )

# #  Add missing columns with NaN
# for col in ["FFV", "Density", "Rg"]:
#     if col not in extra:
#         extra[col] = np.nan

# #  Assign IDs and set index
# next_id = train.index.max() + 1
# extra = extra.set_index(pd.RangeIndex(next_id, next_id + len(extra), name="id"))

# #  Reorder columns to match original data
# extra = extra[train.columns]

# #  Combine with original data
# overall_train = pd.concat([train, extra])

# # Process SMILES
# print("\nGenerating canonical SMILES...")
# overall_train['canonical_smiles'] = overall_train['SMILES'].apply(Config.get_canonical_smiles)
# test['canonical_smiles'] = test['SMILES'].apply(Config.get_canonical_smiles)

# # Verification
# print(f"Original rows: {len(train)}")
# print(f"New rows added: {len(extra)}")
# print(f"Total rows: {len(overall_train)}")


extra = pd.read_csv("/kaggle/input/smile-data/SMILES_EXTRA_DATA (1).csv")

#  Assign IDs and set index
next_id = train.index.max() + 1
extra = extra.set_index(pd.RangeIndex(next_id, next_id + len(extra), name="id"))

#  Reorder columns to match original data
extra = extra[train.columns]

#  Combine with original data
overall_train = pd.concat([train, extra])

# Process SMILES
print("\nGenerating canonical SMILES...")
overall_train['canonical_smiles'] = overall_train['SMILES'].apply(Config.get_canonical_smiles)
test['canonical_smiles'] = test['SMILES'].apply(Config.get_canonical_smiles)

# Verification
print(f"Original rows: {len(train)}")
print(f"New rows added: {len(extra)}")
print(f"Total rows: {len(overall_train)}")


def generate_polymer_features(df, smiles_column='canonical_smiles', radius=2, n_bits=1024):
    def _calculate_features(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            # 1. Calculate traditional descriptors
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
            
            # 2. Calculate Morgan fingerprints
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            fp_features = {f'morgan_{i}': int(bit) for i, bit in enumerate(fp)}
            
            # Combine both feature types
            return {**descriptors, **fp_features}
            
        except Exception as e:
            print(f"Skipping SMILES due to error: {smiles[:50]}... Error: {str(e)[:100]}...")
            return None

    features = []
    valid_indices = []
    
    print(f"Generating features (RDKit descriptors + {n_bits}-bit Morgan fingerprints)...")
    for idx, smi in tqdm(enumerate(df[smiles_column]), total=len(df)):
        feat = _calculate_features(smi)
        if feat is not None:
            features.append(feat)
            valid_indices.append(idx)
    
    features_df = pd.DataFrame(features)
    result_df = pd.concat([
        df.iloc[valid_indices].reset_index(drop=True),
        features_df.reset_index(drop=True)
    ], axis=1)
    
    print(f"Generated {len(features_df.columns)} total features")
    print(f"  - {14} RDKit descriptors")
    print(f"  - {n_bits} Morgan fingerprint bits")
    
    return result_df

train = generate_polymer_features(overall_train)

print("\nğŸ�¯ Generating molecular features for test data...")
test_data = generate_polymer_features(test)

# Identify feature columns
feature_columns = [col for col in train.columns 
                  if col not in Config.TARGET_COLUMNS + ['SMILES','canonical_smiles'] and not col.startswith('id')]

# print("\nğŸ�¯ Feature columns:", feature_columns)


def get_xgb_params(Config):
    """Get base XGBoost parameters"""
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'seed': Config.RANDOM_STATE,
        'verbosity': 0
    }
    
    if Config.USE_GPU:
        params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': Config.GPU_ID,
            'predictor': 'gpu_predictor'
        })
    return params
    

def objective(trial, X, y, groups, Config):
    """Optuna optimization objective function"""
    params = get_xgb_params(Config)
    params.update({
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 2.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    })

    group_kfold = GroupKFold(n_splits=Config.N_FOLDS)
    cv_scores = []

    for train_idx, valid_idx in group_kfold.split(X, y, groups=groups):
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dvalid = xgb.DMatrix(X[valid_idx], label=y[valid_idx])

        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=Config.MAX_ITERATIONS,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=Config.EARLY_STOPPING,
            verbose_eval=False
        )
        cv_scores.append(bst.best_score)

    return np.mean(cv_scores)



def train_single_target(X, y, groups, target_name, Config):
    """Train model for a single target property"""
    print(f"\nStarting optimization for {target_name}")
    
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, X, y, groups, Config),
        n_trials=Config.OPTUNA_TRIALS,
        show_progress_bar=True
    )

    best_params = get_xgb_params(Config)
    best_params.update(study.best_params)
    
    print(f"Best MAE for {target_name}: {study.best_value:.4f}")
    
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=Config.MAX_ITERATIONS,
        verbose_eval=Config.VERBOSE_EVAL
    )

    # Evaluate on training set
    preds = model.predict(dtrain)
    train_mae = mean_absolute_error(y, preds)
    
    return model, {
        'best_cv_mae': study.best_value,
        'train_mae': train_mae,
        'params': best_params,
        'n_samples': len(y)
    }



# Prepare training data
X_train = train[feature_columns].values
y_df = train[Config.TARGET_COLUMNS]
groups = train['canonical_smiles'].factorize()[0]

# Initialize storage for models and results
models = {}
results = {}

# Ensure models directory exists
Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Train models for each target
for target in Config.TARGET_COLUMNS:
    if target not in y_df.columns:
        print(f"Warning: Target {target} not found in data")
        continue
        
    mask = y_df[target].notna()
    if mask.sum() == 0:
        print(f"Warning: No valid samples for target {target}")
        continue
    
    X_target = X_train[mask]
    y_target = y_df[target][mask].values
    groups_target = groups[mask]

    try:
        model, target_results = train_single_target(
            X_target, y_target, groups_target, target, Config
        )
        
        models[target] = model
        results[target] = target_results
        
        # Save model
        model_path = Config.MODEL_DIR / f"{target}_model.json"
        model.save_model(str(model_path))
        
        print(f"âœ… {target}: CV MAE={target_results['best_cv_mae']:.4f}, Train MAE={target_results['train_mae']:.4f}")
        
    except Exception as e:
        print(f"Failed to train {target}: {e}")
        continue

# Print summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

for target, result in results.items():
    print(f"\nğŸ�¯ {target}:")
    print(f"  Samples: {result['n_samples']}")
    print(f"  CV MAE: {result['best_cv_mae']:.4f}")
    print(f"  Train MAE: {result['train_mae']:.4f}")


# Get same feature columns used in training
X_test = test_data[feature_columns].values
print(f"Test data shape: {X_test.shape}")

def generate_predictions(models, X_test, test_ids):
    """Generate predictions for all targets using trained models"""
    predictions = {}
    
    for target, model in models.items():
        try:
            dtest = xgb.DMatrix(X_test)
            predictions[target] = model.predict(dtest)
            print(f"Generated predictions for {target}")
        except Exception as e:
            print(f"Prediction failed for {target}: {e}")
            predictions[target] = np.zeros(len(X_test))  # Default to zeros if prediction fails
    
    # Create DataFrame with IDs and predictions
    submission = pd.DataFrame({'id': test_ids})
    for target in Config.TARGET_COLUMNS:
        if target in predictions:
            submission[target] = predictions[target]
        else:
            print(f"Warning: No predictions for {target}, filling with zeros")
            submission[target] = 0.0
    
    # Ensure correct column order
    submission = submission[['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
    return submission

# Generate predictions
print("\nGenerating predictions...")
submission = generate_predictions(models, X_test, test['id'])

# Save submission file
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission file saved to {submission_path}")

# Show sample of predictions
print("\nSample predictions:")
print(submission.head())



