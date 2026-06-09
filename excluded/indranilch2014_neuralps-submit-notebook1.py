# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
submission.head()
import warnings

warnings.filterwarnings('ignore')


test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test.head()


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train.head()


train.info()


test.info()


submission.info()


#List the Missing Percentage (in each column)for the number of missing values left
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Percentage' :train_na})
missing_data.head()


#Enlist the statistics (mean, minimum and maximum ) for every numerical column

train.describe().loc[['mean','min','max']].T


import matplotlib.pyplot as plt
numerical_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train[numerical_cols].hist(figsize=(15,10), bins=30, color='Magenta', edgecolor='black')
plt.suptitle("Histogram of Numeric Features")
plt.show()


import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(train[numerical_cols].corr(), annot=True, cmap='summer')
plt.title("Correlation Between Numerical Features")
plt.show()


train[numerical_cols].count()


submission.describe()


train.describe()


#Row 7864nd

train.iloc[7864, 1]


train['len'] = [len(i) if i is not None else 0 for i in train['SMILES']]
smiles_lens = [len(i) if i is not None else 0 for i in train['SMILES']]
sns.histplot(smiles_lens)
plt.xlabel('length (smiles)')
plt.ylabel('probability')
plt.title('Smiles Length');


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb


!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

molecules = [
    ('CCO', 'Ethanol - simple alcohol'),
    ('CCCCCCCC', 'Octane - long chain'),
    ('c1ccccc1', 'Benzene - aromatic ring'),
]

for smiles, description in molecules:
    mol = Chem.MolFromSmiles(smiles)
    
    print(f"\n{description}")
    print(f"SMILES: {smiles}")
    print(f"  Molecular Weight: {Descriptors.MolWt(mol):.1f}")
    print(f"  LogP (oiliness): {Descriptors.MolLogP(mol):.2f}")
    print(f"  Rotatable Bonds: {Descriptors.NumRotatableBonds(mol)}")
    print(f"  Aromatic Rings: {Descriptors.NumAromaticRings(mol)}")
    print(f"  Complexity (BertzCT): {Descriptors.BertzCT(mol):.0f}")


train.head(3)


train[numerical_cols].count()


def get_molecular_descriptors(max_autocorr=10):
    """Get molecular descriptors - either hardcoded list or auto-discovered"""

    descriptor_list_all = []
    test_mol = Chem.MolFromSmiles('CCO')

    # Collect all valid descriptors first
    for name in dir(Descriptors):
        if not name.startswith('_'):
            try:
                func = getattr(Descriptors, name)
                if callable(func):
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
    return descriptor_list

molecular_descriptors = get_molecular_descriptors(max_autocorr=10) 


def smiles_to_features(smiles_list, molecular_descriptors, clean_descriptors=False):
    features = []
    total = len(smiles_list)

    print(f"Processing {total} SMILES...", end="", flush=True)

    for i, smiles in enumerate(smiles_list):
        if i > 0 and (i % 1000 == 0 or i == total - 1):
            print(f" {i+1}/{total}", end="", flush=True)

        mol_features = []
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                mol_features = [np.nan] * len(molecular_descriptors)
            else:
                for name, func in molecular_descriptors:
                    try:
                        value = func(mol)
                        if np.isinf(value) or abs(value) > 1e10:
                            value = np.nan
                        mol_features.append(value)
                    except:
                        mol_features.append(np.nan)
        except:
            mol_features = [np.nan] * len(molecular_descriptors)

        features.append(mol_features)

    features = np.array(features, dtype=float)
    print(" âœ…", flush=True)

    if clean_descriptors:
        # % of NaNs per column
        nan_ratio = np.isnan(features).mean(axis=0)
        dropped_mask = nan_ratio > 0.98

        if np.any(dropped_mask):
            dropped_names = [molecular_descriptors[i][0] for i, drop in enumerate(dropped_mask) if drop]
            print(f"âš ï¸� Dropping {len(dropped_names)} descriptors with >98% missing values:")
            print("   " + ", ".join(dropped_names))

            features = features[:, ~dropped_mask]
            molecular_descriptors = [d for i, d in enumerate(molecular_descriptors) if not dropped_mask[i]]

    return features, molecular_descriptors

X_raw, molecular_descriptors = smiles_to_features(train['SMILES'].values, molecular_descriptors, clean_descriptors=True)




def clean_features(X, feature_names=None):
    """Handle NaN/inf values and impute missing data, with concise missing summary."""
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    total_missing = np.isnan(X_clean).sum()
    print(f"ğŸ§¹ Cleaned {total_missing:,} missing values ({total_missing / X_clean.size:.1%})")

    for i in range(X_clean.shape[1]):
        col = X_clean[:, i]
        if np.isnan(col).any():
            missing_pct = np.isnan(col).mean() * 100
            name = feature_names[i] if feature_names else f"col_{i}"
            print(f"   â€¢ {name}: {missing_pct:.1f}% missing")
            median = np.nanmedian(col)
            X_clean[np.isnan(col), i] = median if not np.isnan(median) else 0

    return X_clean

X = clean_features(X_raw)


def train_target_property(X_target, y_target):
    print(f"ğŸ“Š Training on {len(y_target)} samples ")
    print(f"ğŸ“ˆ Target range: {y_target.min():.4f} to {y_target.max():.4f}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_target)
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 127,           
        'learning_rate': 0.07,       
        'feature_fraction': 0.8,     
        'bagging_fraction': 0.9, 
        'bagging_freq': 1,           # Bag every iteration
        'lambda_l1': 0.1,            # L1 regularization
        'lambda_l2': 0.1,            # L2 regularization
        'min_data_in_leaf': 10,      # Prevent overfitting
        'verbose': -1,
        'random_state': 42
    }
    
    # 5-fold cross-validation
    cv_scores = []
    models = []
    all_val_true = []
    all_val_pred = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_target[train_idx], y_target[val_idx]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val)
        cv_score = mean_absolute_error(y_val, val_pred)
        cv_scores.append(cv_score)
        models.append(model)
        
        # Store for competition metric calculation
        all_val_true.extend(y_val)
        all_val_pred.extend(val_pred)
        
        print(f"----Fold {fold+1} Complete / MAE = {cv_score:.4f}", flush=True)
    
    cv_mean = np.mean(cv_scores)
    print(f"===CV: {cv_mean:.4f} Â± {np.std(cv_scores):.3f}===")
    return models, scaler, cv_mean, all_val_true, all_val_pred


# Store trained models and scalers
trained_models = {}
trained_scalers = {}
cv_scores = []

# Store all predictions for final competition score
all_cv_predictions = {}
all_cv_true = {}

# Train each target
for target in numerical_cols:
    print(f"Training {target}...")
    
    # Prepare training data for the current target
    mask = train[target].notna()
    X_target = X[mask]
    y_target = train[target].values[mask]
    models, scaler, cv_score, val_true, val_pred = train_target_property(X_target, y_target)
    trained_models[target] = models
    trained_scalers[target] = scaler
    cv_scores.append(cv_score)
    
    # Store for overall competition score
    all_cv_true[target] = val_true
    all_cv_predictions[target] = val_pred
    print()





# Import the competition metric
import open_polymer_2025_metric as metric

# Create DataFrames for final competition score calculation
cv_true_df = pd.DataFrame()
cv_pred_df = pd.DataFrame()

for target in numerical_cols:
    # Pad shorter arrays with NULL_FOR_SUBMISSION to make them same length
    max_len = max(len(all_cv_true[t]) for t in numerical_cols)
    
    true_padded = list(all_cv_true[target]) + [metric.NULL_FOR_SUBMISSION] * (max_len - len(all_cv_true[target]))
    pred_padded = list(all_cv_predictions[target]) + [metric.NULL_FOR_SUBMISSION] * (max_len - len(all_cv_predictions[target]))
    
    cv_true_df[target] = true_padded
    cv_pred_df[target] = pred_padded

# Add dummy id column
cv_true_df['id'] = range(len(cv_true_df))
cv_pred_df['id'] = range(len(cv_pred_df))

# Calculate individual competition scores
competition_scores = []
for target in numerical_cols:
    comp_score = metric.scaling_error(np.array(all_cv_true[target]), np.array(all_cv_predictions[target]), target)
    competition_scores.append(comp_score)

# Calculate overall competition score
estimated_lb_score = metric.score(cv_true_df, cv_pred_df, 'id')

print("=" * 50)
print(f"Trained: {len(numerical_cols)} targets Ã— 5 CV folds = {len(numerical_cols) * 5} models")
print(f"Average CV MAE across all targets: {np.mean(cv_scores):.4f}")
print(f"Individual competition scores: {[f'{s:.4f}' for s in competition_scores]}")
print(f"ğŸ�¯ ESTIMATED LB SCORE: {estimated_lb_score:.4f}")
print("=" * 50)


def predict_target_property(test, target_name, models, molecular_descriptors, scaler):
    
    print(f"PREDICTING: {target_name}")
    
    if models is None or scaler is None:
        print(f"â�Œ No trained model available for {target_name}, returning zeros")
        return np.zeros(len(test))
    
    # Get molecular features - step by step
    X_raw, _ = smiles_to_features(test['SMILES'].values, molecular_descriptors, clean_descriptors=False)
    X = clean_features(X_raw)
    
    # Scale features using same scaler from training
    X_scaled = scaler.transform(X)
    # Average predictions from all CV folds
    fold_predictions = []
    for model in models:
        pred = model.predict(X_scaled)
        fold_predictions.append(pred)
    
    predictions = np.mean(fold_predictions, axis=0)
    print(f"ğŸ“Š Predictions range: {predictions.min():.4f} to {predictions.max():.4f}")
    
    return predictions


# Load test data
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

print(f"\nMAKING PREDICTIONS...")
all_predictions = {}
for target in numerical_cols:
    predictions = predict_target_property(
        test, target, trained_models[target], molecular_descriptors=molecular_descriptors, scaler=trained_scalers[target]
    )
    all_predictions[target] = predictions

# Create submission
submission = pd.DataFrame({'id': test['id']})
for target in numerical_cols:
    submission[target] = all_predictions[target]

submission.to_csv('submission.csv', index=False)

print(f"Predicted: {len(test)} test samples")
print(f"Saved: submission.csv")

print(f"\nğŸ‘€ SUBMISSION PREVIEW:")
print(submission.head().to_string(index=False, float_format='%.4f'))


submission.describe()


submission.info()

