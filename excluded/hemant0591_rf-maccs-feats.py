!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,StratifiedGroupKFold,GroupKFold,train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles(train['SMILES'][0])
Draw.MolToImage(mol)


train.shape


train.head()


train.isnull().sum()


train.describe(include="all")


missing = train.isnull().sum()[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]
missing


def compute_rdkit_feats(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = {}
    
    maccs = MACCSkeys.GenMACCSKeys(mol)
    
    for i in range(1, maccs.GetNumBits()):  # bit 0 is always 1
        features[f'MACCS_{i}'] = int(maccs.GetBit(i))

    # Descriptors
    descs = {
        'MolWt': Descriptors.MolWt(mol),
        'TPSA': Descriptors.TPSA(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'NumHeavyAtoms': Descriptors.HeavyAtomCount(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumRotatableBonds': CalcNumRotatableBonds(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'MolMR': Descriptors.MolMR(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumHDonors': Descriptors.NumHDonors(mol)
    }
    features.update(descs)

    return features


maccs_data = []

for smile in tqdm(train['SMILES']):
    feats = compute_rdkit_feats(smile)
    if feats is None:
        maccs_data.append(None)
    else:
        maccs_data.append(feats)
        
maccs_df = pd.DataFrame(maccs_data)
train_rdkit = pd.concat([train, maccs_df], axis=1)


train_rdkit.head()


rdkit_cols = train_rdkit.columns.tolist()[7:]
targets = train_rdkit.columns.tolist()[2:7]


models = {}
preds = {}
maes = {}
true_vals = {}
ranges = {}

for target in targets:
    print(f'training model for {target}')
    df = train_rdkit[rdkit_cols + [target]].dropna()
    X = df[rdkit_cols]
    y = df[target]
    
    ranges[target] = y.max() - y.min()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    #predict and save
    y_pred = model.predict(X_val)
    preds[target] = y_pred
    true_vals[target] = y_val.values
    maes[target] = mean_absolute_error(y_val, y_pred)
    
    models[target] = model
    print(f"MAE: {maes[target]:.4f}")


# Count available samples (n_i) for each target
n_samples = {t: len(true_vals[t]) for t in targets}
K = len(targets)

# compute w_i per target
weights = {}
sqrt_inv = [np.sqrt(1/n_samples[t]) for t in targets]
normalizer = K / sum(sqrt_inv)

for t in targets:
    r_i = ranges[t]
    w_i = (1 / r_i) * (normalizer * np.sqrt(1 / n_samples[t]))
    weights[t] = w_i
    
# final weighted mae
wmae = sum(weights[t] * maes[t] for t in targets)


wmae


print("\n Per-target MAEs:")
for t in targets:
    print(f"{t:<10} MAE: {maes[t]:.4f} | Weight: {weights[t]:.4f}")

print(f"\n Final Weighted MAE (Competition Metric): {wmae:.4f}")


maccs_data_test = []

for smile in tqdm(test['SMILES']):
    feats = compute_rdkit_feats(smile)
    if feats is None:
        maccs_data_test.append(None)
    else:
        maccs_data_test.append(feats)
        
maccs_df_test = pd.DataFrame(maccs_data_test)
test_rdkit = pd.concat([test, maccs_df_test], axis=1)


test_rdkit.head()


X_test = test_rdkit[rdkit_cols]

for target in targets:
    print(f'predicting for {target}')
    model = models[target]
    y_pred = model.predict(X_test)
    test_rdkit[target] = y_pred

submission_df = test_rdkit[['id'] + targets]


submission_df


submission_df.to_csv('submission.csv', index=False)

