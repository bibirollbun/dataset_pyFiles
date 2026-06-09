# Install Autogluon for offline use
!pip install -q autogluon --no-index --find-links=file:///kaggle/input/autogluon-install-notebook


# install RDKit for offline use
!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


# List of models supported by autogluon. Select the most relevant ones

hyperparameters = {
    'RF': {},               # Random Forest
    'XT': {},               # Extra Trees
    'KNN': {},              # K-Nearest Neighbors
    'GBM': {},              # LightGBM
    'CAT': {},              # CatBoost
    'NN_TORCH': {},         # Torch Tabular Neural Net
    'LR': {},               # Linear Regression
    'FASTAI': {},           # FastAI Tabular
    'AG_TEXT_NN': {},       # Text neural nets
    'AG_IMAGE_NN': {},      # Image neural nets
    'AG_AUTOMM': {},        # Multi-modal transformer
    'FT_TRANSFORMER': {},   # Tabular transformers
    'TABPFN': {},           # Transformer-trained few-shot predictor
    'TABPFNMIX': {},        # Mixed version of TabPFN
    'FASTTEXT': {},         # Lightweight text model
    'ENS_WEIGHTED': {},     # Weighted ensemble
    'SIMPLE_ENS_WEIGHTED': {},  # Simple ensemble
    'IM_RULEFIT': {},       # InterpretML: rule-based model
    'IM_GREEDYTREE': {},    # InterpretML: greedy tree
    'IM_FIGS': {},          # InterpretML: functional trees
    'IM_HSTREE': {},        # InterpretML: histogram tree
    'IM_BOOSTEDRULES': {},  # InterpretML: boosted rules
}


hyperparameters = {
    'GBM': {}, 'CAT': {}, 'NN_TORCH': {},
    'RF': {}, 'XT': {}, 'LR': {},
    'FASTAI': {}, 'FT_TRANSFORMER': {},
    'ENS_WEIGHTED': {}, 'SIMPLE_ENS_WEIGHTED': {}
}


import os
def is_interactive_session():
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE','') == 'Interactive'

is_interactive_session()

config = {
    "autogluon_time": 60*60*0.2,
    "autogluon_presets": "best_quality",
    #"reduce_features": 0, # Set to >0 to use only the first n features
    "tail_rows": 0 # Set to >0 to use only the last n rows in the file
    
}

if is_interactive_session():
    print("Interactive session")
    config["autogluon_time"] = 100
    #config["reduce_features"] = 200
    config["autogluon_presets"] = "medium_quality"
    config["tail_rows"] = 2000
    print(config)
else:
    print("running as job")
    print(config)


import pandas as pd
import autogluon as ag
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


molecules = [
    ('CCO', 'Ethanol - simple alcohol'),
    ('CCCCCCCC', 'Octane - long chain'),
    ('c1ccccc1', 'Benzene - aromatic ring'),
    ('COO', 'CO2'),
    ("O", "Water")

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
train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train_df.head()


from rdkit.Chem import Descriptors
from rdkit import Chem
import numpy as np

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
   X[np.isinf(X)] = np.nan
   
   # Count and report missing values
   missing = np.isnan(X).sum()
   print(f"ğŸ§¹ Cleaned {missing:,} missing values ({missing/X.size*100:.1f}%)")
   
   # Median imputation
   for i in range(X.shape[1]):
       col = X[:, i]
       if np.isnan(col).any():
           X[np.isnan(col), i] = np.nanmedian(col) if not np.isnan(np.nanmedian(col)) else 0
   
   return X

train_features = pd.DataFrame(clean_features(X_raw))





train_features.columns = feature_names
train_features.head()


train_targets = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]  # Y targets



from autogluon.tabular import TabularPredictor

def train_target_property_autogluon(X, train_df, target_name, time_limit=300, presets="best_quality", hyperparameters={}):
    """
    Trains an AutoGluon model to predict a single target property.

    Returns:
        predictor: Trained TabularPredictor.
        scaler: None (for compatibility with legacy unpacking).
        feature_names: List of feature names used.
        best_model_score: MAE on internal validation.
        leaderboard_df: AutoGluon leaderboard DataFrame.
    """
    # Filter samples with target value
    mask = train_df[target_name].notna()
    X_target = X.loc[mask].copy()
    y_target = train_df.loc[mask, target_name].copy()

    print(f"ğŸ“Š Training on {len(y_target)} samples with target = '{target_name}'")
    print(f"ğŸ“ˆ Target range: {y_target.min():.4f} to {y_target.max():.4f}")

    # Prepare training data
    train_data = X_target.copy()
    train_data[target_name] = y_target
    feature_names = list(X_target.columns)

    # Train
    predictor = TabularPredictor(label=target_name, eval_metric='mae').fit(
        train_data,
        time_limit=time_limit,
        presets=presets,
        hyperparameters = hyperparameters
    )

    # Leaderboard
    leaderboard_df = predictor.leaderboard(silent=True)
    best_model_score = leaderboard_df.loc[0, 'score_val']
    print(f"âœ… Best AutoGluon model MAE: {best_model_score:.4f}")

    return predictor, None, feature_names, best_model_score, leaderboard_df

# Define all target properties
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Store trained models and scalers
trained_models = {}
trained_scalers = {}  # will remain None
cv_scores = []
leaderboards = {}





for target in targets:
    print(f"Training {target}...")
    model, scaler, features, cv_score, lb = train_target_property_autogluon(
        train_features, train_df, target,
        time_limit=config["autogluon_time"],
        presets=config["autogluon_presets"],
        hyperparameters=hyperparameters
    )
    trained_models[target] = model
    trained_scalers[target] = scaler  # remains None
    cv_scores.append(cv_score)
    leaderboards[target] = lb
    print()


import matplotlib.pyplot as plt

for target, lb in leaderboards.items():
    plt.figure()
    plt.title(f"Leaderboard: {target}")
    plt.bar(lb['model'], lb['score_val'])
    plt.xticks(rotation=90)
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.show()



# # Define all target properties
# targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# # Store trained models and scalers
# trained_models = {}
# trained_scalers = {}
# cv_scores = []

# # Train each target - collect results for summary
# for target in targets:
#     print(f"Training {target}...")
#     models, scaler, features, cv_score = train_target_property_autogluon(train_features, train_df, target)
#     trained_models[target] = models
#     trained_scalers[target] = scaler
#     cv_scores.append(cv_score)
#     print()

# # Clean summary with average
# print("=" * 40)
# print(f"Trained: {len(targets)} targets Ã— 5 CV folds = {len(targets) * 5} models")
# print(f"Average CV MAE across all targets: {np.mean(cv_scores):.4f}")


def predict_target_property_autogluon(test_df, target_name, predictor):
    print(f"PREDICTING: {target_name}")
    
    if predictor is None:
        print(f"â�Œ No trained predictor available for {target_name}, returning zeros")
        return np.zeros(len(test_df))
    
    # Make sure test_df is processed to match training features
    descriptor_functions, feature_names = molecular_descriptors
    X_raw = smiles_to_features(test_df['SMILES'].values, descriptor_functions)
    X = pd.DataFrame(clean_features(X_raw))
    X.columns = feature_names 
    
    # AutoGluon works directly with DataFrames
    predictions = predictor.predict(X).values
    print(f"ğŸ“Š Predictions range: {predictions.min():.4f} to {predictions.max():.4f}")
    
    return predictions



print(f"\nMAKING PREDICTIONS...")
all_predictions = {}
for target in targets:
    predictions = predict_target_property_autogluon(
        test_df, target, trained_models[target]
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

