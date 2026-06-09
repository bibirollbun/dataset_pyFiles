!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import os
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']


train[targets].count()


train.describe()


extra_tc_df = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')

display(extra_tc_df.head())


extra_tc_clean = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean':'Tc'})
extra_tc_clean['id'] = range(len(train), len(train) + len(extra_tc_df))
extra_tc_clean[['Tg', 'FFV', 'Density', 'Rg']] = float('nan')

extra_tc_clean = extra_tc_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# æ‹¼æ�¥å��ï¼Œä¸ºç»“æ�œ DataFrame ç”Ÿæˆ�è¿�ç»­çš„æ•´æ•°ç´¢å¼•ï¼Œä»� 0 å¼€å§‹é€’å¢�
train = pd.concat([train, extra_tc_clean], ignore_index=True)

train.count()


def get_molecular_descriptors(max_autocorr=10):
    """Get molecular descriptors - either hardcoded list or auto-discovered"""

    descriptor_list_all = []
    test_mol = Chem.MolFromSmiles('CCO')

    # Collect all valid descriptors first
    for name in dir(Descriptors):  # è¿”å›� Descriptors æ¨¡å�—ä¸‹æ‰€æœ‰çš„å±�æ€§å��ç§°åˆ—è¡¨ï¼ˆå­—ç¬¦ä¸²ï¼‰ï¼ŒåŒ…æ‹¬ï¼šæ��è¿°ç¬¦å‡½æ•°å��ï¼ˆå¦‚ MolWt, MolLogP, TPSA ç­‰ï¼‰ã€�å¸¸é‡�ã€�ç±»ã€�æ–¹æ³•ç­‰
        if not name.startswith('_'):
            try:
                func = getattr(Descriptors, name)  # ä»� Descriptors æ¨¡å�—ä¸­æ ¹æ�®å­—ç¬¦ä¸²å�˜é‡� name è�·å�–å®�é™…çš„å‡½æ•°/å¯¹è±¡ã€‚
                if callable(func):  # åˆ¤æ–­ func æ˜¯å�¦ä¸ºå�¯è°ƒç”¨å¯¹è±¡ï¼ˆå‡½æ•°ã€�æ–¹æ³•ç­‰ï¼‰
                    result = func(test_mol)
                    if isinstance(result, (int, float)) and not np.isnan(result):
                        descriptor_list_all.append((name, func))
            except:
                pass

    print(f"ğŸ”� Total discovered descriptors before filtering: {len(descriptor_list_all)}")

    # Sort AUTOCORR2D descriptors by their numeric suffix
    autocorr_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if name.startswith('AUTOCORR2D_')
    ]
    autocorr_descriptors.sort(key=lambda x: int(x[0].split('_')[-1]))

    # Select only the lowest-numbered ones
    limited_autocorr = autocorr_descriptors[:max_autocorr]

    # Include all other descriptors
    other_descriptors = [
        (name, func)
        for name, func in descriptor_list_all
        if not name.startswith('AUTOCORR2D_')
    ]

    # Final descriptor list
    descriptor_list = limited_autocorr + other_descriptors

    print(f"âœ… Auto-discovered {len(descriptor_list)} descriptors (limited to {max_autocorr} AUTOCORR2D):")
    names = [name for name, _ in descriptor_list]
    print("  " + ", ".join(names))

    feature_names = [name for name, _ in descriptor_list]
    return descriptor_list, feature_names

molecular_descriptors =  get_molecular_descriptors(max_autocorr=10)


def smiles_to_features(smiles_list, descriptor_functions):
   """ Convert SMILES strings to raw feature matrix
   https://www.kaggle.com/code/richolson/smiles-rdkit-lgbm-ftw """

   features = []
   total = len(smiles_list)

   print(f"Processing {total} SMILES...", end="", flush=True)

   for i, smiles in enumerate(smiles_list):
       # Progress indicator every 1000 molecules or at milestones
       if i > 0 and (i % 1000 == 0 or i == total - 1):
           print(f" {i+1}/{total}", end="", flush=True)

       mol_features = []
       try:
           mol = Chem.MolFromSmiles(smiles)
           if mol is None:
               # Invalid SMILES - fill with NaN
               mol_features = [np.nan] * len(descriptor_functions)
           else:
               # Calculate each descriptor
               for name, func in descriptor_functions:
                   try:
                       value = func(mol)
                       # Handle problematic values
                       if np.isinf(value) or abs(value) > 1e10:
                           value = np.nan
                       mol_features.append(value)
                   except:
                       # Descriptor calculation failed
                       mol_features.append(np.nan)
       except:
           # Complete failure - fill entire row with NaN
           mol_features = [np.nan] * len(descriptor_functions)

       features.append(mol_features)

   print(" âœ…", flush=True)
   return np.array(features, dtype=float)

descriptor_functions, feature_names = molecular_descriptors
X_raw = smiles_to_features(train['SMILES'].values, descriptor_functions)
test_raw = smiles_to_features(test['SMILES'].values, descriptor_functions)


def clean_features(X):
    """Handle NaN/inf values and impute missing data"""
    # Create a copy to avoid modifying the original
    X_clean = X.copy()

    X_clean[np.isinf(X_clean)] = np.nan

    # Count and report missing values
    missing = np.isnan(X_clean).sum()
    print(f"ğŸ§¹ Cleaned {missing:,} missing values ({missing/X_clean.size*100:.1f}%)")

    # Median imputation
    for i in range(X_clean.shape[1]):
        col = X_clean[:, i]
        if np.isnan(col).any():
            X_clean[np.isnan(col), i] = np.nanmedian(col) if not np.isnan(np.nanmedian(col)) else 0

    return X_clean


remove_idx_dict = {}
remove_idx_list = []
for target in targets:
    mask = train[target].notna()
    X = X_raw[mask]
    for i in range(X.shape[-1]):
        col = X[:, i]
        nan_ratio = np.isnan(col).sum() / len(col)
        if nan_ratio > 0.8:
            remove_idx_list.append(i)
    remove_idx_dict[target] = remove_idx_list


def xgb_train_predict(X, y):
    params = {
        "Tg":
             {"objective":"reg:squarederror",
              "booster":"gbtree",
              'num_boost_round': 288.0,
              'colsample_bynode': 0.78,
              'colsample_bytree': 0.9,
              'eta': 0.045,
              'gamma': 645.0,
              'lambda': 12.6,
              'max_depth': int(18),
              'min_child_weight': 20.0,
              'subsample': 0.78,
              'seed': 123},
        "FFV":
             {"objective":"reg:absoluteerror",
              "booster":"gbtree",
              'colsample_bynode': 0.42,
              'colsample_bytree': 0.92,
              'eta': 0.045,
              'gamma': 0.052,
              'lambda': 0.8,
              'max_depth': int(20),
              'min_child_weight': 5.0,
              'num_boost_round': 268,
              'subsample': 0.94,
              'seed': 123},
        "Tc":
             {"objective":"reg:absoluteerror",
              "booster":"gbtree",
              'colsample_bynode': 0.72,
              'colsample_bytree': 0.655,
              'eta': 0.21,
              'gamma': 0.28,
              'lambda': 12.5,
              'max_depth': int(17),
              'min_child_weight': 1.8,
              'num_boost_round': 221,
              'subsample': 0.95,
              'seed': 123},
        "Density":
             {"objective":"reg:absoluteerror",
              "booster":"gbtree",
              'colsample_bynode': 0.72,
              'colsample_bytree': 0.72,
              'eta': 0.09,
              'gamma': 0.33,
              'lambda': 1.27,
              'max_depth': int(23),
              'min_child_weight': 3.7,
              'num_boost_round': 190,
              'subsample': 0.9,
              'seed': 123},
        "Rg":
             {"objective":"reg:squarederror",
              "booster":"gbtree",
              'colsample_bynode': 0.72,
              'colsample_bytree': 0.5,
              'eta': 0.065,
              'gamma': 2.0,
              'lambda': 5.75,
              'max_depth': 10,
              'min_child_weight': 4,
              'num_boost_round': 124,
              'subsample': 0.88,
              'seed': 123},
    }
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-fold cross-validation
    cv_scores = []
    models = []
    all_val_true = []
    all_val_pred = []

    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        model = xgb.train(params[target], dtrain, num_boost_round=int(params[target]['num_boost_round']))
        val_pred = model.predict(dval)

        cv_score = mean_absolute_error(y_val, val_pred)
        cv_scores.append(cv_score)
        models.append(model)

        all_val_true.extend(y_val)
        all_val_pred.extend(val_pred)

        print(f"----Fold {fold+1} Complete / MAE = {cv_score:.4f}", flush=True)

    cv_mean = np.mean(cv_scores)
    print(f"===CV: {cv_mean:.4f} Â± {np.std(cv_scores):.3f}===")

    return models, scaler, cv_mean, all_val_true, all_val_pred


%%time
import warnings

# Store trained models and scalers
trained_models = {}
trained_scalers = {}
cv_scores = []

# Store all predictions for final competition score
all_cv_predictions = {}
all_cv_true = {}

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    for target in targets:
        print(f"Training {target}...")
        mask = train[target].notna()
        X = X_raw[mask]
        X = np.delete(X, remove_idx_dict[target], axis=1)
        print(f"Removed {len(remove_idx_dict[target])} features with {target}")
        X = clean_features(X)
        y = train[target][mask].values
        if np.isnan(X).any():
            print(f"NaN values found in X with {target}")

        models, scaler, cv_score, val_true, val_pred = xgb_train_predict(X, y)
        trained_models[target] = models
        trained_scalers[target] = scaler
        cv_scores.append(cv_score)

        all_cv_true[target] = val_true
        all_cv_predictions[target] = val_pred


def xgb_test_predict(test_df, target, models, scaler):
    print(f"PREDICTING: {target}")

    if models is None or scaler is None:
        print(f"â�Œ No trained model available for {target}, returning zeros")
        return np.zeros(len(test_df))
    test_scaled = scaler.transform(test_df)
    test_xgb = xgb.DMatrix(test_scaled)

    fold_predictions = []
    for model in models:
        fold_pred = model.predict(test_xgb)
        fold_predictions.append(fold_pred)
    predictions = np.mean(fold_predictions, axis=0)
    print(f"ğŸ“Š Predictions range: {predictions.min():.4f} to {predictions.max():.4f}")
    return predictions


all_predictions = {}
for target in targets:
    X_test = np.delete(test_raw, remove_idx_dict[target], axis=1)
    X_test = clean_features(X_test)
    if np.isnan(X_test).any():
        print(f"NaN values found in X_test with {target}")

    predictions = xgb_test_predict(X_test, target, trained_models[target], trained_scalers[target])
    all_predictions[target] = predictions

submission = pd.DataFrame({'id': test['id']})
for target in targets:
    submission[target] = all_predictions[target]

submission.to_csv('submission.csv', index=False)


submission

