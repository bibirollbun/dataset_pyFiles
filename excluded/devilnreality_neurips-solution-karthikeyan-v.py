import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')

# Load training data
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# Display shape and head
print("Train shape:", train.shape)
train.head()


# Target columns
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Plot missing value ratio per target
train[target_cols].isnull().mean().sort_values(ascending=False).plot(
    kind='barh', figsize=(8, 4), color='teal'
)
plt.title('Missing Value Ratio per Property')
plt.xlabel('Missing Ratio')
plt.grid(True)
plt.show()


# Plot value distributions for available targets
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, col in enumerate(target_cols):
    sns.histplot(train[col], bins=30, ax=axes[i], kde=True, color='slateblue')
    axes[i].set_title(col)

plt.tight_layout()
plt.show()


# Correlation matrix between targets (where labels are present)
corr = train[target_cols].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Between Target Properties')
plt.show()


# Install RDKit and compatible NumPy version
!pip install -q rdkit-pypi
!pip install -q numpy==1.24.4


from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


# Get list of descriptor names
descriptor_list = [desc[0] for desc in Descriptors._descList]

# Compute descriptor vector for a single SMILES
def compute_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(descriptor_list)
    descriptors = []
    for _, desc in Descriptors._descList:
        try:
            descriptors.append(desc(mol))
        except:
            descriptors.append(np.nan)
    return descriptors

# Apply to train set
X_rdkit = train['SMILES'].apply(compute_rdkit_descriptors)
rdkit_df = pd.DataFrame(X_rdkit.tolist(), columns=descriptor_list)

# Quick check
print("RDKit descriptor shape:", rdkit_df.shape)
rdkit_df.head()


from rdkit.Chem import AllChem

# Function to compute 2048-bit Morgan fingerprints
def morgan_fp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)

# Apply to all SMILES in train
X_fp = np.stack(train['SMILES'].apply(morgan_fp))
fp_df = pd.DataFrame(X_fp)

# Sanity check
print("Morgan fingerprint shape:", fp_df.shape)
fp_df.head()


from sklearn.metrics import mean_absolute_error
import numpy as np

# Estimate weights from training data
range_dict, inv_sqrt_counts, denom = {}, {}, 0
for t in target_cols:
    vals = train[t].dropna()
    r = vals.max() - vals.min()
    n = len(vals)
    range_dict[t] = r
    inv_sqrt_counts[t] = 1 / np.sqrt(n)
    denom += inv_sqrt_counts[t]

K = len(target_cols)
weights = {
    t: (1 / range_dict[t]) * (K * inv_sqrt_counts[t] / denom)
    for t in target_cols
}
print("Estimated leaderboard-style weights:\n", weights)

# Custom weighted MAE scorer
def weighted_mae(y_true_dict, y_pred_dict, weights):
    score = 0
    for target in weights:
        mask = y_true_dict[target].notnull()
        if mask.sum() == 0:
            continue
        mae = mean_absolute_error(
            y_true_dict[target][mask],
            y_pred_dict[target][mask]
        )
        score += weights[target] * mae
    return score


from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# ----------------------------
# Combine feature sets
# ----------------------------
X = pd.concat([rdkit_df, fp_df], axis=1)

# ----------------------------
# Define models for each target
# ----------------------------
model_config = {
    'Tg': CatBoostRegressor(verbose=0, random_state=42),
    'Density': CatBoostRegressor(verbose=0, random_state=42),
    'FFV': XGBRegressor(n_estimators=100, random_state=42, missing=np.inf),
    'Tc': XGBRegressor(n_estimators=100, random_state=42, missing=np.inf),
    'Rg': LGBMRegressor(n_estimators=200, random_state=42)
}

target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# ----------------------------
# Estimate competition-style weights
# ----------------------------
range_dict, inv_sqrt_counts, denom = {}, {}, 0
for t in target_cols:
    vals = train[t].dropna()
    r = vals.max() - vals.min()
    n = len(vals)
    range_dict[t] = r
    inv_sqrt_counts[t] = 1 / np.sqrt(n)
    denom += inv_sqrt_counts[t]

K = len(target_cols)
weights = {
    t: (1 / range_dict[t]) * (K * inv_sqrt_counts[t] / denom)
    for t in target_cols
}
print("ğŸ“� Estimated target weights:", weights)

# ----------------------------
# Define custom weighted MAE
# ----------------------------
def weighted_mae(y_true_dict, y_pred_dict, weights):
    score = 0
    for t in weights:
        y_true = y_true_dict[t]
        y_pred = y_pred_dict[t]
        mask = y_true.notnull()
        if mask.sum() == 0:
            continue
        score += weights[t] * mean_absolute_error(y_true[mask], y_pred[mask])
    return score

# ----------------------------
# Model training loop
# ----------------------------
best_models = {}
target_mae = {}

for target in target_cols:
    y = train[target]
    idx = y.notnull()
    X_sub, y_sub = X.loc[idx].copy(), y[idx].copy()

    # Replace infs for numerical safety
    if np.isinf(X_sub.values).sum() > 0:
        print(f"\nâš ï¸� Replacing infs in features for target: {target}")
        X_sub.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Skip if not enough samples
    if len(X_sub) < 5:
        print(f"\nğŸš« Skipping {target} (only {len(X_sub)} samples)")
        continue

    model = model_config[target]
    print(f"\nğŸ�¯ Training model for: {target} [{model.__class__.__name__}]")

    scores = cross_val_score(model, X_sub, y_sub, scoring='neg_mean_absolute_error', cv=5)
    mae = -scores.mean()
    print(f"ğŸ“Š CV MAE ({target}): {mae:.4f}")

    model.fit(X_sub, y_sub)
    best_models[target] = model
    target_mae[target] = mae

print("\nâœ… Training complete.")
print("ğŸ“Œ MAE per property:")
print(target_mae)


# Load test data
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

# RDKit descriptors
X_rdkit_test = test['SMILES'].apply(compute_rdkit_descriptors)
rdkit_test_df = pd.DataFrame(X_rdkit_test.tolist(), columns=rdkit_df.columns)

# Morgan fingerprints
fp_array_test = np.stack(test['SMILES'].apply(morgan_fp))
fp_test_df = pd.DataFrame(fp_array_test, columns=fp_df.columns)

# Combine feature sets
X_test = pd.concat([rdkit_test_df, fp_test_df], axis=1)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)


# Initialize prediction DataFrame
predictions = pd.DataFrame({'SMILES': test['SMILES']})

# Generate predictions using trained models
for target in best_models:
    print(f"ğŸ”® Predicting {target}...")
    preds = best_models[target].predict(X_test)
    predictions[target] = preds

# Sanity check
predictions.head()


# Load sample submission to preserve format
sample_sub = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')

# Overwrite prediction columns
sample_sub[target_cols] = predictions[target_cols]

# Save to submission.csv
sample_sub.to_csv('submission.csv', index=False)
print("âœ… submission.csv created successfully.")

