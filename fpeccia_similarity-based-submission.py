!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import numpy as np

morgan_gen = GetMorganGenerator(fpSize=2048)


train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


train_smiles_canon = []
for train_smile in tqdm(train_df["SMILES"]):
    cs = Chem.CanonSmiles(train_smile)
    train_smiles_canon.append(cs)
train_df["SMILES_canon"] = train_smiles_canon

train_mols = [Chem.MolFromSmiles(x) for x in tqdm(train_df["SMILES_canon"])]
train_fps = [morgan_gen.GetFingerprint(x) for x in tqdm(train_mols)]


test_smiles_canon = []
for test_smile in tqdm(test_df["SMILES"]):
    cs = Chem.CanonSmiles(test_smile)
    test_smiles_canon.append(cs)
test_df["SMILES_canon"] = test_smiles_canon
test_mols = [Chem.MolFromSmiles(x) for x in tqdm(test_df["SMILES_canon"])]
test_fps = [morgan_gen.GetFingerprint(x) for x in tqdm(test_mols)]


predictions = {
    'Tg': [],
    'FFV': [],
    'Tc': [],
    'Density': [],
    'Rg': []
}
for test_fp in test_fps:
    #similarities = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkAsymmetricSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkBraunBlanquetSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkCosineSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkDiceSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkKulczynskiSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkMcConnaugheySimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkOnBitSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkRogotGoldbergSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkRusselSimilarity(test_fp, train_fps)
    similarities = DataStructs.BulkSokalSimilarity(test_fp, train_fps)
    #similarities = DataStructs.BulkTverskySimilarity(test_fp, train_fps)
    most_similar_train_smiles = train_smiles_canon[np.argmax(similarities)]
    match = train_df[train_df.SMILES_canon == most_similar_train_smiles]
    predictions['Tg'].append(match['Tg'].values[0])
    predictions['FFV'].append(match['FFV'].values[0])
    predictions['Tc'].append(match['Tc'].values[0])
    predictions['Density'].append(match['Density'].values[0])
    predictions['Rg'].append(match['Rg'].values[0])


submission = pd.DataFrame({
    'id': test_df['id'].values,  # Ensure we're using values
    **predictions
})
    
# Check for any missing values
missing_values = submission.isnull().sum()
if missing_values.any():
    # Fill missing values with median of each column
    for col in submission.columns:
        if submission[col].isnull().any():
            median_val = submission[col].median()
            submission[col] = submission[col].fillna(median_val)
            print(f"Filled missing values in {col} with median: {median_val}")


# Ensure all columns are in the correct order
expected_columns = ['id', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
submission = submission[expected_columns]
    
# Save submission
submission.to_csv('submission.csv', index=False)


import seaborn as sns
import matplotlib.pyplot as plt


data = {
    'Similarity Type': [
        'Tanimoto', 
        'Asymmetric', 
        'BraunBlanquet',
        'Cosine',
        'Dice',
        'Kulczynski',
        'McConnaughey',
        'OnBit',
        'RogotGoldberg',
        'Russel',
        'Sokal',
        #'Tversky',
        'Tanimoto', 
        'Asymmetric', 
        'BraunBlanquet',
        'Cosine',
        'Dice',
        'Kulczynski',
        'McConnaughey',
        'OnBit',
        'RogotGoldberg',
        'Russel',
        'Sokal',
        #'Tversky'
    ],
    'Morgan Fingerprint Size': [
        1024, 
        1024, 
        1024, 
        1024, 
        1024, 
        1024, 
        1024, 
        1024, 
        1024, 
        1024, 
        1024, 
        #1024,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        2048,
        #2048
    ],
    'LB': [
        0.136, 
        0.193, 
        0.135,
        0.135,
        0.136,
        0.134,
        0.134,
        0.136,
        0.136,
        0.142,
        0.136,
        #0.,
        0.136,
        0.186,
        0.135,
        0.136,
        0.136,
        0.134,
        0.134,
        0.136,
        0.136,
        0.138,
        0.136,
        #0.
    ]
}

# Create the DataFrame
df = pd.DataFrame(data)


fig, ax = plt.subplots(1,1, figsize=(16,4))
fig.suptitle('LB scores results by predicting properties based on different similarity metrics.')
sns.barplot(df, x="Similarity Type", y="LB", hue="Morgan Fingerprint Size", ax=ax)
ax.set_ylim([0.130, 0.200])
plt.show()

