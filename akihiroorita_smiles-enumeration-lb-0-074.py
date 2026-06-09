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
from sklearn.model_selection import train_test_split

import pandas as pd

csv_path = '/kaggle/input/neurips-open-polymer-prediction-2025/train.csv'
train_df = pd.read_csv(csv_path)
csv2_path = '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv'
test_df = pd.read_csv(csv2_path)


# install RDKit for offline use
!pip install /kaggle/input/rdkit-install-whl/rdkit_wheel/rdkit_pypi-2022.9.5-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


train_df.shape


from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import pandas as pd
from tqdm import tqdm

# 1. 208å€‹ã�®descriptorå��ã‚’å�–å¾—
descriptor_names = [desc[0] for desc in Descriptors._descList]

# 2. è¨ˆç®—å™¨ã‚’æº–å‚™
calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

# 3. è¨˜è¿°å­�ã‚’è¨ˆç®—ã�™ã‚‹é–¢æ•°
def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(descriptor_names)
    return calc.CalcDescriptors(mol)

# 4. SMILESåˆ—ã�‹ã‚‰descriptorã‚’è¨ˆç®—
descriptor_df = pd.DataFrame(
    [compute_descriptors(smi) for smi in tqdm(train_df['SMILES'])],
    columns=descriptor_names
)

# 5. å…ƒã�®train_dfã�¨é€£çµ�ï¼ˆã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æ�ƒã�ˆï¼‰
train_df_with_desc = pd.concat([train_df.reset_index(drop=True), descriptor_df], axis=1)

# Optional: æ¬ æ��ã‚’ç¢ºèª�
print(train_df_with_desc.isnull().sum().sort_values(ascending=False).head())


from rdkit import Chem
from rdkit.Chem import MolToSmiles
import random

def enumerate_smiles(smiles, n_aug=5, seed=42):
    """SMILESã‚’ãƒ©ãƒ³ãƒ€ãƒ ã�«ä¸¦ã�³æ›¿ã�ˆã�¦è¤‡æ•°ç”Ÿæˆ�"""
    random.seed(seed)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [MolToSmiles(mol, doRandom=True) for _ in range(n_aug)]


train_df = train_df_with_desc


# Tg, Tc, Density, Rgã�«ã�¤ã�„ã�¦å…¨ã�¦æ‹¡å¼µ

import pandas as pd

targets = ['Tg', 'Tc', 'Density', 'Rg']
n_aug = 5  # æ‹¡å¼µæ•°

augmented_dfs = []

for target in targets:
    print(f"ğŸ”� æ‹¡å¼µä¸­: {target}")
    df_target = train_df[train_df[target].notnull()].copy()

    augmented_rows = []

    for _, row in df_target.iterrows():
        smiles = row['SMILES']
        for enum_smiles in enumerate_smiles(smiles, n_aug=n_aug):
            new_row = row.copy()
            new_row['SMILES'] = enum_smiles
            augmented_rows.append(new_row)

    augmented_df = pd.DataFrame(augmented_rows)
    augmented_dfs.append(augmented_df)

# å…¨ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã�®æ‹¡å¼µçµ�æ�œã‚’çµ�å�ˆï¼ˆå…ƒãƒ‡ãƒ¼ã‚¿ã�¯é‡�è¤‡ã�—ã�ªã�„ã�®ã�§å®‰å…¨ï¼‰
augmented_all_df = pd.concat(augmented_dfs, ignore_index=True)

# å…ƒã�®train_dfã�¨çµ�å�ˆã�—ã�¦æ‹¡å¼µå®Œäº†
train_df_augmented = pd.concat([train_df, augmented_all_df], ignore_index=True)


train_df_augmented


from lightgbm import LGBMRegressor

for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    target_series = train_df_augmented[target]  # â†� ä¿®æ­£
    non_null_mask = target_series.notnull()

    X_train = train_df_augmented.loc[non_null_mask, descriptor_names]  # â†� ä¿®æ­£
    y_train = target_series.loc[non_null_mask]

    X_pred = train_df_augmented.loc[~non_null_mask, descriptor_names]  # â†� ä¿®æ­£

    if X_pred.shape[0] == 0:
        print(f"No missing values in {target}")
        continue

    model = LGBMRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_pred)

    train_df_augmented.loc[~non_null_mask, target] = y_pred  # â†� ä¿®æ­£
    print(f"Filled missing values in {target} with predicted values")


# test_df ã�«è¨˜è¿°å­�ã‚’è¿½åŠ 
descriptor_test_df = pd.DataFrame(
    [compute_descriptors(smi) for smi in tqdm(test_df['SMILES'])],
    columns=descriptor_names
)
test_df_with_desc = pd.concat([test_df.reset_index(drop=True), descriptor_test_df], axis=1)


from lightgbm import LGBMRegressor

models = {}  # â†� ãƒ¢ãƒ‡ãƒ«ã‚’ä¿�å­˜ã�™ã‚‹è¾�æ›¸

for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    target_series = train_df_augmented[target]
    non_null_mask = target_series.notnull()

    X_train = train_df_augmented.loc[non_null_mask, descriptor_names]
    y_train = target_series.loc[non_null_mask]

    X_pred = train_df_augmented.loc[~non_null_mask, descriptor_names]

    if X_pred.shape[0] == 0:
        print(f"No missing values in {target}")
    else:
        print(f"Filled missing values in {target} with predicted values")

    model = LGBMRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)
    
    # æ¬ æ��ã�Œã�‚ã‚Œã�°äºˆæ¸¬ã�—ã�¦è£œå®Œ
    if X_pred.shape[0] > 0:
        y_pred = model.predict(X_pred)
        train_df_augmented.loc[~non_null_mask, target] = y_pred

    models[target] = model  # â†� ã�“ã�“ã�§ä¿�å­˜


# testç”¨äºˆæ¸¬é–¢æ•°
def predict_test(df, model, target_col, feature_cols):
    df[target_col] = model.predict(df[feature_cols])
    return df

# FFV äºˆæ¸¬ï¼ˆdescriptorã�®ã�¿ï¼‰
test_df_with_desc = predict_test(test_df_with_desc, models['FFV'], 'FFV', descriptor_names)

# test_df_with_desc ã�«å¯¾ã�—ã�¦å…¨ã‚¿ãƒ¼ã‚²ãƒƒãƒˆã‚’äºˆæ¸¬
for target in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    test_df_with_desc = predict_test(test_df_with_desc, models[target], target, descriptor_names)

# æ��å‡ºç”¨ãƒ•ã‚¡ã‚¤ãƒ«ã�®ä½œæˆ�
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
submission = test_df_with_desc[['id'] + targets].copy()
submission.to_csv('submission.csv', index=False)

print("âœ… æ��å‡ºç”¨ãƒ•ã‚¡ã‚¤ãƒ« 'submission.csv' ã‚’ä½œæˆ�ã�—ã�¾ã�—ã�Ÿã€‚")
print(submission.head())




