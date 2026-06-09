pip install -q rdkit


!pip install -q autogluon.tabular scikit-learn==1.5.2


import time
import shutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Descriptors
from autogluon.tabular import TabularPredictor
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint

# Ignore all warnings to keep the output clean
warnings.filterwarnings("ignore")

# Start timing the script
start_time = time.time()

print("Loading data...")
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the data is in the correct directory.")
    exit()

# Identify features and target
TARGET = "BeatsPerMinute"
ID_COL = "id"
SMILES_COL = "TrackDurationMs"

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


train_df.columns


# ------------------ Feature Engineering & Imputation ------------------ #
def wrangle(df):
    # Columns to check for 1.07e-06
    columns_to_impute = [
        'RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
        'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
        'TrackDurationMs', 'Energy'
    ]
    
    # Impute 1.07e-06 with mean for numerical columns
    for col in columns_to_impute:
        if col in df.columns:
            # Replace 1.07e-06 with NaN, then impute with mean
            df[col] = df[col].replace(1.07e-06, np.nan)
            df[col] = df[col].fillna(df[col].mean())
    
    # Log transformation for skewed numerical features
    df['TrackDurationMs_log'] = np.log1p(df['TrackDurationMs'])
    
    # Interaction features
    df['Energy_MoodScore'] = df['Energy'] * df['MoodScore']
    df['Rhythm_Acoustic'] = df['RhythmScore'] * df['AcousticQuality']
    df['Vocal_Instrumental'] = df['VocalContent'] * df['InstrumentalScore']
      # Binary features
    df['HighEnergy'] = (df['Energy'] > df['Energy'].median()).astype(int)
    df['HighLiveLikelihood'] = (df['LivePerformanceLikelihood'] > df['LivePerformanceLikelihood'].median()).astype(int)
    
    # Normalize duration to minutes
    df['TrackDurationMin'] = df['TrackDurationMs'] / (1000 * 60)
    
    # New feature: Ratio of VocalContent to InstrumentalScore
    df['Vocal_to_Instrumental'] = df['VocalContent'] / (df['InstrumentalScore'] + 1e-6)  # Avoid division by zero
    
    # Drop original columns to avoid redundancy
    df = df.drop(columns=['TrackDurationMs'], errors='ignore')
    
    return df

train_df = wrangle(train_df)
test_df  = wrangle(test_df)
SMILES_COL = "TrackDurationMs_log"


def generate_features_from_smiles(smiles_list):
    """
    Generates all features for a list of SMILES strings using RDKit.
    Returns a list of feature dictionaries.
    """
    features = []
    for smiles in tqdm(smiles_list, desc="Processing SMILES"):
        feature_dict = {}

        # âœ… Handle NaN or non-string SMILES
        if not isinstance(smiles, str) or smiles.strip() == "":
            features.append(feature_dict)  # empty dict, will become NaN later
            continue

        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            # Descriptors
            for name, func in Descriptors.descList:
                try:
                    feature_dict[name] = func(mol)
                except Exception:
                    feature_dict[name] = np.nan
            
            # Morgan Fingerprint
            fp_morgan = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            for i in range(fp_morgan.GetNumBits()):
                feature_dict[f"Morgan_{i}"] = int(fp_morgan.GetBit(i))
            
            # MACCS Keys
            fp_maccs = GetMACCSKeysFingerprint(mol)
            for i in range(fp_maccs.GetNumBits()):
                feature_dict[f"MACCS_{i}"] = int(fp_maccs.GetBit(i))

        features.append(feature_dict)
    return features
# Parallelize feature generation to speed up the process
def parallel_feature_generation(smiles_list, num_workers):
    """Splits the SMILES list and generates features in parallel."""
    chunks = np.array_split(smiles_list, num_workers)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(generate_features_from_smiles, chunks)
    
    # Flatten the list of lists into a single list
    return [item for sublist in results for item in sublist]


# Get the number of CPU cores to use for parallel processing
num_cores = multiprocessing.cpu_count()
print(f"Using {num_cores} CPU cores for parallel feature generation.")

# Generate features for both training and test sets
train_features_list = parallel_feature_generation(train_df[SMILES_COL].tolist(), num_cores)
test_features_list = parallel_feature_generation(test_df[SMILES_COL].tolist(), num_cores)

train_rdkit_features = pd.DataFrame(train_features_list).fillna(0)
test_rdkit_features = pd.DataFrame(test_features_list).fillna(0)

# Combine RDKit features with existing features and the target
train_ag = pd.concat([train_df.drop(columns=[SMILES_COL, ID_COL]), train_rdkit_features], axis=1)
test_ag = pd.concat([test_df.drop(columns=[SMILES_COL, ID_COL]), test_rdkit_features], axis=1)



print(f"Shape of training data after RDKit feature generation: {train_ag.shape}")
print(f"Shape of test data after RDKit feature generation: {test_ag.shape}")
print(f"Number of total features: {test_ag.shape[1]}")


predictor = TabularPredictor(
    label=TARGET,
    problem_type='regression',
    eval_metric='rmse'
).fit(
    train_data=train_ag,
    time_limit=3600*9,
    presets='best_quality' 
)


predictor.leaderboard(silent=True).style.background_gradient(subset=['score_val'], cmap='RdYlGn') 


print("\nGenerating test predictions...")
test_preds = predictor.predict(test_ag) 


submission = sample_submission.copy()
submission[TARGET] = test_preds
submission.to_csv("submission.csv", index=False)
print("\nSubmission file 'submission.csv' created successfully.")

# End timing the script
end_time = time.time()
print(f"\nTotal script execution time: {end_time - start_time:.2f} seconds")

submission.head() 


shutil.rmtree("AutogluonModels") 

