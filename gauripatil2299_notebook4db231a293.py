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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load datasets
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
sample_submission = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


# Preview data
print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())
display(test.head())


# Check for missing values
print("\nMissing values in train:")
print(train.isnull().sum())


# Summary statistics of target columns
target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
train[target_cols].describe()


# Plot distributions of targets
plt.figure(figsize=(15, 8))
for i, col in enumerate(target_cols):
    plt.subplot(2, 3, i + 1)
    sns.histplot(train[col], kde=True, bins=50)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


from rdkit import Chem
from rdkit.Chem import Descriptors


# Function to convert SMILES to RDKit molecule and calculate descriptors
def featurize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * 5
    return [
        Descriptors.MolWt(mol),               # Molecular weight
        Descriptors.MolLogP(mol),             # LogP (hydrophobicity)
        Descriptors.TPSA(mol),                # Topological Polar Surface Area
        Descriptors.NumRotatableBonds(mol),   # Flexibility
        Descriptors.RingCount(mol)            # Ring count
    ]


# Apply to train data
feature_names = ['MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'RingCount']
train_features = train['SMILES'].apply(featurize_smiles)
train_feat_df = pd.DataFrame(train_features.tolist(), columns=feature_names)


# Combine with target (e.g., FFV only for now)
ffv_train = pd.concat([train_feat_df, train['FFV']], axis=1).dropna()
print(ffv_train.head())


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Features & Target
X = ffv_train[feature_names]
y = ffv_train['FFV']


# Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


from lightgbm import early_stopping, log_evaluation

# Model
model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    learning_rate=0.05
)

# Train the model with callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=20)]
)

# Predict and evaluate
val_preds = model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print(f"\nValidation MAE (FFV): {mae:.6f}")



# Featurize test set
test_features = test['SMILES'].apply(featurize_smiles)
test_feat_df = pd.DataFrame(test_features.tolist(), columns=feature_names)

# Predict FFV on test set
test_preds = model.predict(test_feat_df)

# Create submission DataFrame
submission = test[['id']].copy()
submission['Tg'] = 0.0  # placeholder
submission['FFV'] = test_preds
submission['Tc'] = 0.0
submission['Density'] = 0.0
submission['Rg'] = 0.0

# Save to CSV
submission.to_csv('submission.csv', index=False)
submission.head()



# ğŸ“Œ INSTALL RDKit (Internet ON)
!pip install -q rdkit-pypi

# ğŸ“¦ IMPORTS
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# ğŸ“‚ Load data
train = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")

# ğŸ‘©â€�ğŸ”¬ Define molecular descriptors you want to extract
descriptor_funcs = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "RotatableBonds": Descriptors.NumRotatableBonds,
    "RingCount": Descriptors.RingCount
}

# ğŸ”� Function to convert SMILES â†’ RDKit Mol â†’ Features
def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {name: None for name in descriptor_funcs}
    return {name: func(mol) for name, func in descriptor_funcs.items()}

# ğŸ§ª Convert SMILES to features for train/test
train_features = train["SMILES"].apply(smiles_to_features).apply(pd.Series)
test_features = test["SMILES"].apply(smiles_to_features).apply(pd.Series)

# ğŸ”¢ Add ID columns back
train_features.insert(0, "id", train["id"])
test_features.insert(0, "id", test["id"])

# ğŸ’¾ Save to CSV (as output files to download)
train_features.to_csv("rdkit_train_features.csv", index=False)
test_features.to_csv("rdkit_test_features.csv", index=False)


