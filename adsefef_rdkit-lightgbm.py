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


# Step 1: Load and explore the training data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
train_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
df = pd.read_csv(train_path)

# Basic info
print("ğŸ”� Basic Info:")
print(df.info())
print("\nğŸ§¾ Sample Rows:")
print(df.head())

# Missing values
print("\nâ�“ Missing Values:")
print(df.isnull().sum())

# Summary statistics of target properties
print("\nğŸ“Š Descriptive Statistics (Targets):")
print(df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].describe())

# Distribution plots for each target property
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
for col in target_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, bins=50)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# SMILES string stats
df['smiles_length'] = df['SMILES'].apply(len)
print("\nğŸ§ª SMILES Length Stats:")
print(df['smiles_length'].describe())

plt.figure()
sns.histplot(df['smiles_length'], bins=50, kde=True)
plt.title('Distribution of SMILES String Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()



# Count the number of unique SMILES strings
num_unique_smiles = df['SMILES'].nunique()
total_smiles = len(df)

print(f"ğŸ”¢ Total rows in dataset: {total_smiles}")
print(f"ğŸ§¬ Unique SMILES strings: {num_unique_smiles}")
print(f"ğŸ“Š Fraction of unique SMILES: {num_unique_smiles / total_smiles:.2%}")



# âœ… Install RDKit in Kaggle
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem



# STEP 2: Feature Extraction using RDKit Descriptors

# ğŸ”§ Install RDKit if needed (Kaggle usually has it pre-installed)
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
import pandas as pd
from tqdm import tqdm

# ğŸ‘‡ This is your training DataFrame from earlier
# df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# Step 1: Get all available descriptor names
descriptor_names = [desc_name for desc_name, _ in Descriptors._descList]

# Step 2: Function to compute all descriptors for one molecule
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_names)
    return [desc(mol) for _, desc in Descriptors._descList]

# Step 3: Compute descriptors for all SMILES
tqdm.pandas()
descriptor_data = df['SMILES'].progress_apply(compute_descriptors)

# Step 4: Convert to DataFrame
descriptor_df = pd.DataFrame(descriptor_data.tolist(), columns=descriptor_names)
print(f"âœ… Shape of descriptor matrix: {descriptor_df.shape}")

# Step 5: Combine with target columns
data_with_features = pd.concat([df[['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']], descriptor_df], axis=1)

# Optional: Save to CSV for quick reloads
# data_with_features.to_csv('polymer_descriptors.csv', index=False)

# Show sample
data_with_features.head()



from sklearn.model_selection import train_test_split

# Step 1: Drop descriptor columns with too many NaNs (e.g., >25%)
nan_threshold = 0.25
valid_cols = descriptor_df.columns[descriptor_df.isnull().mean() < nan_threshold]
clean_descriptor_df = descriptor_df[valid_cols]

print(f"âœ… Retained {len(valid_cols)} descriptors out of {descriptor_df.shape[1]}")

# Step 2: Drop rows with any remaining NaNs in features
final_df = pd.concat([df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']], clean_descriptor_df], axis=1)
final_df = final_df.dropna(subset=clean_descriptor_df.columns)  # drop bad SMILES / incomplete features

# Step 3: Keep track of SMILES and ID if you need them later
ids = df.loc[final_df.index, 'id']
smiles = df.loc[final_df.index, 'SMILES']

# Step 4: Features and Targets
X = final_df.drop(columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
y = final_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]

print(f"ğŸ“� Feature matrix shape: {X.shape}")
print(f"ğŸ�¯ Target matrix shape: {y.shape}")

# Step 5: Train/Validation Split (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("âœ… Train/Val split complete")
print(f"ğŸ§ª Train: {X_train.shape[0]} rows")
print(f"ğŸ§ª  Val:   {X_val.shape[0]} rows")



import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Store models and predictions
models = {}
val_preds = {}
train_scores = {}

for target in y.columns:
    print(f"ğŸ�¯ Training model for: {target}")
    
    # Filter out rows where this target is not missing
    mask = ~y_train[target].isna()
    X_target = X_train[mask]
    y_target = y_train.loc[mask, target]
    
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbosity=-1)
    model.fit(X_target, y_target)
    
    models[target] = model
    
    # Predict on validation set
    val_preds[target] = model.predict(X_val)
    
    # Score only where the ground truth exists
    val_mask = ~y_val[target].isna()
    score = mean_absolute_error(y_val.loc[val_mask, target], val_preds[target][val_mask])
    train_scores[target] = score
    
    print(f"âœ… {target} MAE: {score:.4f}\n")

# Summary of scores
print("ğŸ“Š Validation MAEs:")
for target, score in train_scores.items():
    print(f" - {target}: {score:.4f}")



import numpy as np
import pandas as pd

# Step 1: Build prediction DataFrame aligned to y_val
y_val_preds_df = pd.DataFrame(index=y_val.index, columns=y_val.columns)

for target in y_val.columns:
    print(f"ğŸ”® Predicting for: {target}")
    val_mask = ~y_val[target].isna()
    preds = models[target].predict(X_val[val_mask])
    y_val_preds_df.loc[val_mask, target] = preds

# Step 2: Compute competition-style weighted MAE
def compute_wmae(y_true_df, y_pred_df):
    tasks = y_true_df.columns
    n_samples = {}
    ranges = {}
    inv_sqrt_weights = {}

    for col in tasks:
        mask = ~y_true_df[col].isna()
        n = mask.sum()
        if n == 0:
            continue
        n_samples[col] = n
        value_range = y_true_df[col].max() - y_true_df[col].min()
        ranges[col] = value_range
        inv_sqrt_weights[col] = 1 / np.sqrt(n)

    weight_sum = sum(inv_sqrt_weights.values())
    norm_weights = {k: v / weight_sum for k, v in inv_sqrt_weights.items()}

    wmae = 0
    for col in tasks:
        mask = ~y_true_df[col].isna()
        abs_error = np.abs(y_true_df.loc[mask, col] - y_pred_df.loc[mask, col])
        normalized_error = abs_error / ranges[col]
        task_mae = normalized_error.mean()
        wmae += norm_weights[col] * task_mae

    return wmae

# Step 3: Calculate and print
wmae_score = compute_wmae(y_val, y_val_preds_df)
print(f"ğŸ�¯ Weighted MAE (competition metric): {wmae_score:.5f}")




# Parameters
n_bits = 2048  # Size of fingerprint
radius = 2     # How far to look from each atom

# Convert SMILES to Morgan fingerprint
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

# Apply to all SMILES
tqdm.pandas()
morgan_fps = df['SMILES'].progress_apply(smiles_to_morgan)

# Convert to array (exclude failed SMILES)
valid_mask = ~morgan_fps.isna()
morgan_array = np.stack(morgan_fps[valid_mask].values)
morgan_df = pd.DataFrame(morgan_array, index=df[valid_mask].index)

# Extract aligned targets
y_morgan = df.loc[valid_mask, ['Tg', 'FFV', 'Tc', 'Density', 'Rg']]

print(f"âœ… Morgan feature matrix shape: {morgan_df.shape}")
print(f"ğŸ�¯ Aligned targets shape: {y_morgan.shape}")



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

# Step 1: Split
X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
    morgan_df, y_morgan, test_size=0.2, random_state=42
)

# Step 2: Train per target
models_morgan = {}
val_preds_morgan = {}
for target in y_train_m.columns:
    print(f"ğŸ�¯ Training model for: {target}")
    mask = ~y_train_m[target].isna()
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbosity=-1)
    model.fit(X_train_m[mask], y_train_m.loc[mask, target])
    models_morgan[target] = model

# Step 3: Predict per target
y_val_preds_morgan = pd.DataFrame(index=y_val_m.index, columns=y_val_m.columns)
for target in y_val_m.columns:
    print(f"ğŸ”® Predicting for: {target}")
    mask = ~y_val_m[target].isna()
    preds = models_morgan[target].predict(X_val_m[mask])
    y_val_preds_morgan.loc[mask, target] = preds

# Step 4: Compute weighted MAE
wmae_score_morgan = compute_wmae(y_val_m, y_val_preds_morgan)
print(f"ğŸ�¯ Weighted MAE (Morgan model): {wmae_score_morgan:.5f}")



# Align both feature sets and targets
shared_index = X.index.intersection(morgan_df.index)
X_stacked = pd.concat([X.loc[shared_index], morgan_df.loc[shared_index]], axis=1)
y_stacked = y.loc[shared_index]

print(f"ğŸ“� Combined feature matrix shape: {X_stacked.shape}")
print(f"ğŸ�¯ Target matrix shape: {y_stacked.shape}")



# Split into train/val
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_stacked, y_stacked, test_size=0.2, random_state=42
)

# Train per target
models_stacked = {}
val_preds_stacked = {}

for target in y_train_s.columns:
    print(f"ğŸ�¯ Training stacked model for: {target}")
    mask = ~y_train_s[target].isna()
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, verbosity=-1)
    model.fit(X_train_s[mask], y_train_s.loc[mask, target])
    models_stacked[target] = model

# Predict
y_val_preds_stacked = pd.DataFrame(index=y_val_s.index, columns=y_val_s.columns)

for target in y_val_s.columns:
    print(f"ğŸ”® Predicting for: {target}")
    mask = ~y_val_s[target].isna()
    preds = models_stacked[target].predict(X_val_s[mask])
    y_val_preds_stacked.loc[mask, target] = preds

# Compute wMAE
wmae_score_stacked = compute_wmae(y_val_s, y_val_preds_stacked)
print(f"ğŸ�¯ Weighted MAE (Stacked features): {wmae_score_stacked:.5f}")



# Load test set
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test_ids = test_df['id']
test_smiles = test_df['SMILES']
# --- RDKit Descriptors ---
test_rdkit_feats = test_smiles.progress_apply(compute_descriptors)
# First build full test RDKit frame
full_test_rdkit_df = pd.DataFrame(test_rdkit_feats.tolist(), columns=descriptor_names)

# Then select only the valid columns used in training
test_rdkit_df = full_test_rdkit_df[valid_cols]

# --- Morgan Fingerprints ---
test_morgan_feats = test_smiles.progress_apply(smiles_to_morgan)
test_morgan_array = np.stack(test_morgan_feats.values)
test_morgan_df = pd.DataFrame(test_morgan_array)

# --- Combine RDKit + Morgan ---
X_test_stacked = pd.concat([test_rdkit_df, test_morgan_df], axis=1)



# Predict using stacked models
submission_preds = pd.DataFrame(index=test_ids, columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])

for target in submission_preds.columns:
    print(f"ğŸ§  Predicting {target}")
    model = models_stacked[target]
    submission_preds[target] = model.predict(X_test_stacked)



submission_preds.reset_index(drop=True, inplace=True)
submission_preds.insert(0, 'id', test_ids.values)  # Re-add 'id' cleanly
submission_preds.to_csv('/kaggle/working/submission.csv', index=False)


