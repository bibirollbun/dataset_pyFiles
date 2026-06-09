import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install /kaggle/input/optuna-integration/optuna_integration-3.6.0-py3-none-any.whl


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

sample = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


train.head()


train.info()


test.info()


sample.info()


# --- IMPORTS ---
import pandas as pd
import numpy as np
import warnings

# Modeling
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Hyperparameter Optimization
import optuna

# RDKit for Feature Engineering
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

# Suppress warnings for a cleaner output
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- CONFIGURATION ---
TARGET_VARIABLES = ["Tg", "FFV", "Tc", "Density", "Rg"]
N_SPLITS = 5
RANDOM_STATE = 42
OPTUNA_N_TRIALS = 30 # Increase for better results, but longer runtime.


# --- RDKIT FEATURE ENGINEERING ---
def generate_rdkit_features(smiles_str: str):
    """
    Generates RDKit features using methods compatible with older library versions.
    """
    mol = Chem.MolFromSmiles(smiles_str)
    
    desc_list = [d[0] for d in Descriptors._descList]
    morgan_fp_size = 1024
    
    if mol is None:
        # Return NaNs for invalid SMILES
        total_feature_count = len(desc_list) + morgan_fp_size
        return np.full(total_feature_count, np.nan)

    # 1. Descriptors (using the direct, compatible method)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    descriptors = np.array(calculator.CalcDescriptors(mol))
    
    # 2. Morgan Fingerprints (using the direct, compatible function call)
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=morgan_fp_size)
    mfp_array = np.array(list(mfp.ToBitString())).astype(int)
    
    return np.concatenate([descriptors, mfp_array])


# --- DATA PREPARATION ---
print("Loading data and generating features...")
train_df, test_df, sample_df = train, test, sample

desc_list_names = [d[0] for d in Descriptors._descList]
fp_morgan_cols = [f'mfp_{i}' for i in range(1024)]
feature_columns = desc_list_names + fp_morgan_cols

X = pd.DataFrame(np.vstack([generate_rdkit_features(s) for s in train_df['SMILES']]), columns=feature_columns)
X_test = pd.DataFrame(np.vstack([generate_rdkit_features(s) for s in test_df['SMILES']]), columns=feature_columns)

# Robust Data Cleaning
f32_max = np.finfo(np.float32).max
for df in [X, X_test]:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[df > f32_max] = np.nan
    df[df < -f32_max] = np.nan
    
impute_values = X.mean()
X.fillna(impute_values, inplace=True)
X_test = X_test.reindex(columns=X.columns).fillna(impute_values)
print(f"Feature generation complete. Shape of X: {X.shape}")


# --- OPTUNA OBJECTIVE FUNCTIONS (FINAL CORRECTED VERSION) ---
def create_objective(X, y, model_name):
    def objective(trial):
        if model_name == 'xgb':
            params = {
                'objective': 'reg:squarederror', 'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist'
            }
            model_class = xgb.XGBRegressor
            callback = xgb.callback.EarlyStopping(50, save_best=False)
            fit_params = {'callbacks': [callback], 'verbose': 0}
        else: # lgb
            params = {
                'objective': 'regression_l1', 'n_estimators': 1000,
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1
            }
            model_class = lgb.LGBMRegressor
            callback = lgb.early_stopping(50, verbose=False)
            fit_params = {'callbacks': [callback]} # No 'verbose' for LGBM's .fit() method
        
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = model_class(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_params)
            preds = model.predict(X_val)
            scores.append(np.sqrt(mean_squared_error(y_val, preds)))
        return np.mean(scores)
    return objective

# --- MAIN TRAINING PIPELINE ---
META_MODEL = Ridge(alpha=1.0, random_state=RANDOM_STATE)
final_predictions = pd.DataFrame({'id': test_df['id']})

for target in TARGET_VARIABLES:
    print(f"\n--- Processing Target: {target} ---")
    y = train_df[target].dropna()
    X_subset = X.loc[y.index]

    print(f"  Running Optuna to find best XGBoost params...")
    xgb_objective = create_objective(X_subset, y, 'xgb')
    xgb_study = optuna.create_study(direction='minimize')
    xgb_study.optimize(xgb_objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    best_xgb_params.update({'n_estimators': 2000, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'tree_method': 'hist'})

    print(f"  Running Optuna to find best LightGBM params...")
    lgb_objective = create_objective(X_subset, y, 'lgb')
    lgb_study = optuna.create_study(direction='minimize')
    lgb_study.optimize(lgb_objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=True)
    best_lgb_params = lgb_study.best_params
    best_lgb_params.update({'n_estimators': 2000, 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbosity': -1})

    print("  Training final Stacking models with best params...")
    oof_preds_xgb = pd.Series(np.zeros(len(X_subset)), index=X_subset.index)
    oof_preds_lgb = pd.Series(np.zeros(len(X_subset)), index=X_subset.index)
    test_preds_xgb_folds = np.zeros((len(X_test), N_SPLITS))
    test_preds_lgb_folds = np.zeros((len(X_test), N_SPLITS))
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_subset, y)):
        print(f"    Fold {fold+1}/{N_SPLITS}...")
        X_train_fold, y_train_fold = X_subset.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X_subset.iloc[val_idx], y.iloc[val_idx]
        
        xgb_model = xgb.XGBRegressor(**best_xgb_params)
        xgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[xgb.callback.EarlyStopping(50, save_best=True)], verbose=0)
        oof_preds_xgb.iloc[val_idx] = xgb_model.predict(X_val_fold)
        test_preds_xgb_folds[:, fold] = xgb_model.predict(X_test)

        lgb_model = lgb.LGBMRegressor(**best_lgb_params)
        lgb_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds_lgb.iloc[val_idx] = lgb_model.predict(X_val_fold)
        test_preds_lgb_folds[:, fold] = lgb_model.predict(X_test)
        
    X_meta_train = pd.concat([oof_preds_xgb, oof_preds_lgb], axis=1)
    X_meta_test = pd.DataFrame({'xgb': np.mean(test_preds_xgb_folds, axis=1), 'lgb': np.mean(test_preds_lgb_folds, axis=1)})
    
    meta_model = META_MODEL
    meta_model.fit(X_meta_train, y)
    final_predictions[target] = meta_model.predict(X_meta_test)

# --- SUBMISSION ---
final_predictions = final_predictions[sample_df.columns]
final_predictions.to_csv('submission.csv', index=False)
print("\nSubmission file created using Optuna-tuned Stacking strategy.")

