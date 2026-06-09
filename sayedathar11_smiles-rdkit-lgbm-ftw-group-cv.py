# install RDKit for offline use
!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold # NEW: Import for scaffold generation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold # Use GroupKFold with scaffolds
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import open_polymer_2025_metric as metric # Ensure this is imported correctly for local evaluation
import warnings

warnings.filterwarnings('ignore')


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


# Load data
comp_train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
comp_train_df.head(3)


# Define all target properties
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Count of non-NaN values for each target column
comp_train_df[targets].count()


extra_tg_df = pd.read_csv('/kaggle/input/smiles-tg/Tg_SMILES_class_pid_polyinfo_median.csv')
display(extra_tg_df.head(3))

extra_tc_df = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
display(extra_tc_df.head(3))


# Prepare extra_tg_df
extra_tg_clean = extra_tg_df[['SMILES', 'PID', 'Tg']].rename(columns={'PID': 'id'})
extra_tg_clean[['FFV', 'Tc', 'Density', 'Rg']] = float('nan')

# Prepare extra_tc_df  
extra_tc_clean = extra_tc_df[['SMILES', 'TC_mean']].rename(columns={'TC_mean': 'Tc'})
extra_tc_clean['id'] = range(len(comp_train_df) + len(extra_tg_df), len(comp_train_df) + len(extra_tg_df) + len(extra_tc_df))
extra_tc_clean[['Tg', 'FFV', 'Density', 'Rg']] = float('nan')

# Reorder columns to match train_df
extra_tg_clean = extra_tg_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]
extra_tc_clean = extra_tc_clean[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']]

# Combine all datasets into train_df
train_df = pd.concat([comp_train_df, extra_tg_clean, extra_tc_clean], ignore_index=True)

print(train_df[targets].count())


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
    return descriptor_list, feature_names

molecular_descriptors =  get_molecular_descriptors(max_autocorr=10) 


def smiles_to_features(smiles_list, descriptor_functions):
   """Convert SMILES strings to raw feature matrix"""
   
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
X_raw = smiles_to_features(train_df['SMILES'].values, descriptor_functions)    


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

X = clean_features(X_raw)


# --- REVERTED: Generate groups for GroupKFold using Canonical SMILES ---
# Function to get canonical SMILES
def get_canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol) # Get canonical SMILES (default is canonical)
    except:
        return None



print("\nGenerating canonical SMILES groups for GroupKFold...")
train_df['canonical_smiles'] = train_df['SMILES'].apply(get_canonical_smiles)

# Assign unique group IDs to each unique canonical SMILES
# Ensure that if get_canonical_smiles returns None, it gets its own group.
canonical_smiles_group_map = {s: i for i, s in enumerate(train_df['canonical_smiles'].unique()) if s is not None}
train_groups = train_df['canonical_smiles'].map(canonical_smiles_group_map).values.astype(float) # Use float to handle NaNs

# Handle cases where canonical SMILES generation failed (resulting in NaN in train_groups)
# Assign a unique group ID to NaN canonical SMILES.
nan_smiles_group_id = max(canonical_smiles_group_map.values()) + 1 if canonical_smiles_group_map else 0
# Assign unique IDs for each NaN canonical SMILES
train_groups[np.isnan(train_groups)] = nan_smiles_group_id + np.arange(np.isnan(train_groups).sum()) 
train_groups = train_groups.astype(int) # Convert back to int for GroupKFold

print(f"Generated {len(canonical_smiles_group_map)} unique canonical SMILES groups for GroupKFold.")
# CORRECTED LINE: Count None values directly
print(f"Number of None canonical SMILES: {train_df['canonical_smiles'].isna().sum()} (these were assigned unique groups)")


def train_target_property(X_target, y_target, groups_target): # Added groups_target parameter
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
    
    # 5-fold GroupKFold cross-validation
    cv_scores = []
    models = []
    all_val_true = []
    all_val_pred = []
    
    kf = GroupKFold(n_splits=5) 
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_target, groups=groups_target)):
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
all_cv_predictions = {}
all_cv_true = {}

# Train each target
for target in targets:
    print(f"\nTraining {target}...")
    
    mask = train_df[target].notna()
    X_target = X[mask]
    y_target = train_df[target].values[mask]
    
    # Get corresponding groups for the masked data
    groups_target = train_groups[mask]
    
    models, scaler, cv_score, val_true, val_pred = train_target_property(X_target, y_target, groups_target)
    trained_models[target] = models
    trained_scalers[target] = scaler
    cv_scores.append(cv_score)
    
    all_cv_true[target] = val_true
    all_cv_predictions[target] = val_pred
    print()




# Import the competition metric
import open_polymer_2025_metric as metric

# Create DataFrames for final competition score calculation
cv_true_df = pd.DataFrame()
cv_pred_df = pd.DataFrame()

for target in targets:
    # Pad shorter arrays with NULL_FOR_SUBMISSION to make them same length
    max_len = max(len(all_cv_true[t]) for t in targets)
    
    true_padded = list(all_cv_true[target]) + [metric.NULL_FOR_SUBMISSION] * (max_len - len(all_cv_true[target]))
    pred_padded = list(all_cv_predictions[target]) + [metric.NULL_FOR_SUBMISSION] * (max_len - len(all_cv_predictions[target]))
    
    cv_true_df[target] = true_padded
    cv_pred_df[target] = pred_padded

# Add dummy id column
cv_true_df['id'] = range(len(cv_true_df))
cv_pred_df['id'] = range(len(cv_pred_df))

# Calculate individual competition scores
competition_scores = []
for target in targets:
    comp_score = metric.scaling_error(np.array(all_cv_true[target]), np.array(all_cv_predictions[target]), target)
    competition_scores.append(comp_score)

# Calculate overall competition score
estimated_lb_score = metric.score(cv_true_df, cv_pred_df, 'id')

print("=" * 50)
print(f"Trained: {len(targets)} targets Ã— 5 CV folds = {len(targets) * 5} models")
print(f"Average CV MAE across all targets: {np.mean(cv_scores):.4f}")
print(f"Individual competition scores: {[f'{s:.4f}' for s in competition_scores]}")
print(f"ğŸ�¯ ESTIMATED LB SCORE: {estimated_lb_score:.4f}")
print("=" * 50)


def predict_target_property(test_df, target_name, models, scaler):
    
    print(f"PREDICTING: {target_name}")
    
    if models is None or scaler is None:
        print(f"â�Œ No trained model available for {target_name}, returning zeros")
        return np.zeros(len(test_df))
    
    # Get molecular features - step by step
    descriptor_functions, _ = molecular_descriptors
    X_raw = smiles_to_features(test_df['SMILES'].values, descriptor_functions)
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
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

print(f"\nMAKING PREDICTIONS...")
all_predictions = {}
for target in targets:
    predictions = predict_target_property(
        test_df, target, trained_models[target], trained_scalers[target]
    )
    all_predictions[target] = predictions

# Create submission
submission = pd.DataFrame({'id': test_df['id']})
for target in targets:
    submission[target] = all_predictions[target]

submission.to_csv('submission.csv', index=False)

print(f"Predicted: {len(test_df)} test samples")
print(f"Saved: submission.csv")

print(f"\nğŸ‘€ SUBMISSION PREVIEW:")
print(submission.head().to_string(index=False, float_format='%.4f'))




