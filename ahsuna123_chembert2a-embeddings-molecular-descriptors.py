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


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
t0start = time.time() 

import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pickle


%%time

train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
train.head()


train.info()


%%time

test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test.head()


from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "DeepChem/ChemBERTa-77M-MTR"

# Download and cache locally
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained("./chemberta-77M-MTR")
tokenizer.save_pretrained("./chemberta-77M-MTR")



%%time
N = 1
smiles = train['SMILES'].iat[0]#[:30]
print('Len of smiles string:',len(smiles))
print(smiles)
with torch.no_grad():
    padding=True
    encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
    model_output = chemberta(**encoded_input)
    print()
    print('model_output:')
    print('model_output info - type:', type(model_output),'len(model_output)', len(model_output) ,)
    print('type( model_output[0]):', type( model_output[0]), 'model_output[0].shape:',  model_output[0].shape  )
    print('model_output - first 100 symbols', str( model_output)[:100] )
    print()
    
    embedding = model_output[0][:,0,:]
    embeddings_cls = embedding
    print('embeddings_cls: type', type(embeddings_cls), 'shape:', embeddings_cls.shape)
    print()

    embedding = torch.mean(model_output[0],1)
    embeddings_mean = embedding
    print('embeddings_mean: type', type(embeddings_mean), 'shape:', embeddings_mean.shape)
    print()
    
# print(embeddings_cls)


embedding.shape


model_output[0].shape


%%time
from tqdm import tqdm

chemberta.eval()
def featurize_ChemBERTa(smiles_list, padding=True):
    # Get hidden size dynamically from a dummy input
    sample_input = tokenizer(smiles_list[0], return_tensors="pt", padding=padding, truncation=True)
    with torch.no_grad():
        sample_output = chemberta(**sample_input)
    hidden_dim = sample_output[0].shape[-1]

    embeddings_cls = torch.zeros(len(smiles_list), hidden_dim)
    embeddings_mean = torch.zeros(len(smiles_list), hidden_dim)

    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            try:
                encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
                model_output = chemberta(**encoded_input)
                embeddings_cls[i] = model_output[0][:, 0, :]
                embeddings_mean[i] = torch.mean(model_output[0], 1)
            except Exception as e:
                print(f"Failed on {i}th SMILES: {smiles}. Error: {e}")
    
    return embeddings_cls.numpy(), embeddings_mean.numpy()



%%time
smiles_list = train['SMILES'].to_list()
print(smiles_list[:10])  # Optional: just for preview
train_cls_pad_true, train_mean_pad_true = featurize_ChemBERTa(smiles_list)


%%time
smiles_list_1 = test['SMILES'].to_list()
print(smiles_list_1[:10])  # Optional: just for preview
test_cls_pad_true, test_mean_pad_true = featurize_ChemBERTa(smiles_list_1)


train_cls_pad_true.shape


test_cls_pad_true.shape


%%time
np.save('train_ChemBERTa_v2_77MTR_cls_pad_True.npy', train_cls_pad_true)
np.save('train_ChemBERTa_v2_77MTR_mean_pad_True.npy', train_mean_pad_true)


np.save('test_ChemBERTa_v2_77MTR_cls_pad_True.npy', test_cls_pad_true)
np.save('test_ChemBERTa_v2_77MTR_mean_pad_True.npy', test_mean_pad_true)


%%time

for emb in [train_cls_pad_true, train_mean_pad_true]:
    
    cm = np.corrcoef(emb[:100,:].T)
    cm[np.isnan(cm)] = 0 
    print(cm.shape)
    sns.clustermap(cm, cmap='coolwarm')
    plt.title('Correlation of emb coordinates',fontsize = 20)
    plt.show()


    cm = np.corrcoef(emb[:700,:])
    cm[np.isnan(cm)] = 0 
    print(cm.shape)    
    sns.clustermap(cm, cmap='coolwarm')
    plt.title('Correlations of smiles',fontsize = 20)
    plt.show()


emb = np.asarray(emb)
emb = emb[~np.isnan(emb).any(axis=1)]
emb = emb[~np.all(emb == 0, axis=1)]


%%time
import umap

reducer = umap.UMAP()

for emb in [train_cls_pad_true, train_mean_pad_true]:
    r = reducer.fit_transform(emb)
    sns.scatterplot(x = r[:,0],y = r[:,1])
    plt.show()
    d = pd.DataFrame(r)
    d=d.reset_index()
    display( d.corr() )


import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Fragments, Lipinski
from rdkit.Chem import rdmolops
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'
RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def get_canonical_smiles(smiles):
    if not RDKIT_AVAILABLE:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return smiles
print("ğŸ“‚ Loading competition data...")

train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')
print(f" Training samples: {len(train)}")
print(f" Test samples: {len(test)}")

def clean_and_validate_smiles(smiles):
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]',
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        '([R])', '([R1])', '([R2])',
    ]
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    return smiles
    
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)
invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()

print(f" Removed {invalid_train} invalid SMILES from training data")
print(f" Removed {invalid_test} invalid SMILES from test data")
train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f" Final training samples: {len(train)}")
print(f" Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    print(f" Processing {len(df_extra)} {target} samples...")
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    print(f" Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f" No valid data remaining for {target}")
        return df_train
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = \
                df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)
    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f' {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    return df_train
    
external_datasets = []

def safe_load_dataset(path, target, processor_func, description):
    try:
        if path.endswith('.xlsx'):
            data = pd.read_excel(path)
        else:
            data = pd.read_csv(path)
        data = processor_func(data)
        external_datasets.append((target, data))
        print(f" âœ… {description}: {len(data)} samples")
        return True
    except Exception as e:
        print(f" âš ï¸� {description} failed: {str(e)[:100]}")
        return False
safe_load_dataset(
    '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'Tc',
    lambda df: df.rename(columns={'TC_mean': 'Tc'}),
    'Tc data'
)
safe_load_dataset(
    '/kaggle/input/tgss-backup/TgSS_enriched_cleaned.csv',
    'Tg',
    lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
    'TgSS enriched data'
)
safe_load_dataset(
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'Tg',
    lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
    'JCIM Tg data'
)
safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'Tg',
    lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15),
    'Xlsx Tg data'
)
safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'Density',
    lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
    'Density data'
)
safe_load_dataset(
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv',
    'FFV',
    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
    'dataset 4'
)
print("\nğŸ”„ Integrating external data...")

train_extended = train[['SMILES'] + TARGETS].copy()
for target, dataset in external_datasets:
    print(f" Processing {target} data...")
    train_extended = add_extra_data_clean(train_extended, dataset, target)
print(f"\nğŸ“Š Final training data:")
print(f" Original samples: {len(train)}")
print(f" Extended samples: {len(train_extended)}")
print(f" Gain: +{len(train_extended) - len(train)} samples")

for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f" {target}: {count:,} samples (+{gain})")
print(f"\nâœ… Data integration complete with clean SMILES!")

def separate_subtables(train_df):
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    subtables = {}
    for label in labels:
        subtables[label] = train_df[['SMILES', label]][train_df[label].notna()]
    return subtables
    
def augment_smiles_dataset(smiles_list, labels, num_augments=3):
    augmented_smiles = []
    augmented_labels = []
    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        augmented_smiles.append(smiles)
        augmented_labels.append(label)
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles)
            augmented_labels.append(label)
    return augmented_smiles, np.array(augmented_labels)
    
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.Descriptors import MolWt, MolLogP
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import networkx as nx

def smiles_to_combined_fingerprints_with_descriptors(smiles_list, radius=2, n_bits=128):
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []
    descriptors = []
    valid_smiles = []
    invalid_indices = []
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            morgan_fp = generator.GetFingerprint(mol)
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            combined_fp = np.concatenate([
                np.array(morgan_fp),
                np.array(maccs_fp)
            ])
            fingerprints.append(combined_fp)
            descriptor_values = {}
            for name, func in Descriptors.descList:
                try:
                    descriptor_values[name] = func(mol)
                except:
                    descriptor_values[name] = 0
            descriptor_values['MolWt'] = MolWt(mol)
            descriptor_values['LogP'] = MolLogP(mol)
            descriptor_values['TPSA'] = CalcTPSA(mol)
            descriptor_values['RotatableBonds'] = CalcNumRotatableBonds(mol)
            descriptor_values['NumAtoms'] = mol.GetNumAtoms()
            descriptor_values['SMILES'] = smiles
            try:
                adj = rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)
                if nx.is_connected(G):
                    descriptor_values['graph_diameter'] = nx.diameter(G)
                    descriptor_values['avg_shortest_path'] = nx.average_shortest_path_length(G)
                else:
                    descriptor_values['graph_diameter'] = 0
                    descriptor_values['avg_shortest_path'] = 0
                descriptor_values['num_cycles'] = len(list(nx.cycle_basis(G)))
            except:
                descriptor_values['graph_diameter'] = 0
                descriptor_values['avg_shortest_path'] = 0
                descriptor_values['num_cycles'] = 0
            descriptors.append(descriptor_values)
            valid_smiles.append(smiles)
        else:
            fingerprints.append(np.zeros(n_bits + 167))
            descriptors.append({})
            valid_smiles.append(None)
            invalid_indices.append(i)
    return np.array(fingerprints), descriptors, valid_smiles, invalid_indices
    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}
filters = {
    'Tg': list(set([
        'BalabanJ','BertzCT','Chi1','Chi3n','Chi4n','EState_VSA4','EState_VSA8',
        'FpDensityMorgan3','HallKierAlpha','Kappa3','MaxAbsEStateIndex','MolLogP',
        'NumAmideBonds','NumHeteroatoms','NumHeterocycles','NumRotatableBonds',
        'PEOE_VSA14','Phi','RingCount','SMR_VSA1','SPS','SlogP_VSA1','SlogP_VSA5',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState4','VSA_EState6','VSA_EState7',
        'VSA_EState8','fr_C_O_noCOO','fr_NH1','fr_benzene','fr_bicyclic','fr_ether',
        'fr_unbrch_alkane'
    ]).union(required_descriptors)),
    'FFV': list(set([
        'AvgIpc','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
        'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','EState_VSA10','EState_VSA5',
        'EState_VSA7','EState_VSA8','EState_VSA9','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','FractionCSP3','HallKierAlpha',
        'HeavyAtomMolWt','Kappa1','Kappa2','Kappa3','MaxAbsEStateIndex',
        'MaxEStateIndex','MinEStateIndex','MolLogP','MolMR','MolWt','NHOHCount',
        'NOCount','NumAromaticHeterocycles','NumHAcceptors','NumHDonors',
        'NumHeterocycles','NumRotatableBonds','PEOE_VSA14','RingCount','SMR_VSA1',
        'SMR_VSA10','SMR_VSA3','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA9','SPS',
        'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
        'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',
        'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
        'VSA_EState8','VSA_EState9','fr_Ar_N','fr_C_O','fr_NH0','fr_NH1',
        'fr_aniline','fr_ether','fr_halogen','fr_thiophene'
    ]).union(required_descriptors)),
    'Tc': list(set([
        'BalabanJ','BertzCT','Chi0','EState_VSA5','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HeavyAtomMolWt','MinEStateIndex',
        'MolWt','NumAtomStereoCenters','NumRotatableBonds','NumValenceElectrons',
        'SMR_VSA10','SMR_VSA7','SPS','SlogP_VSA6','SlogP_VSA8','VSA_EState1',
        'VSA_EState7','fr_NH1','fr_ester','fr_halogen'
    ]).union(required_descriptors)),
    'Density': list(set([
        'BalabanJ','Chi3n','Chi3v','Chi4n','EState_VSA1','ExactMolWt',
        'FractionCSP3','HallKierAlpha','Kappa2','MinEStateIndex','MolMR','MolWt',
        'NumAliphaticCarbocycles','NumHAcceptors','NumHeteroatoms',
        'NumRotatableBonds','SMR_VSA10','SMR_VSA5','SlogP_VSA12','SlogP_VSA5',
        'TPSA','VSA_EState10','VSA_EState7','VSA_EState8'
    ]).union(required_descriptors)),
    'Rg': list(set([
        'AvgIpc','Chi0n','Chi1v','Chi2n','Chi3v','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HallKierAlpha','HeavyAtomMolWt',
        'Kappa3','MaxAbsEStateIndex','MolWt','NOCount','NumRotatableBonds',
        'NumUnspecifiedAtomStereoCenters','NumValenceElectrons','PEOE_VSA14',
        'PEOE_VSA6','SMR_VSA1','SMR_VSA5','SPS','SlogP_VSA1','SlogP_VSA2',
        'SlogP_VSA7','SlogP_VSA8','VSA_EState1','VSA_EState8','fr_alkyl_halide',
        'fr_halogen'
    ]).union(required_descriptors))
}

from sklearn.mixture import GaussianMixture
def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")
    X.columns = X.columns.astype(str)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")
    df = X.copy()
    df['Target'] = y.values
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)
    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    X_augmented = augmented_df.drop(columns='Target')
    y_augmented = augmented_df['Target']
    return X_augmented, y_augmented
    
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)
test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
output_df = pd.DataFrame({
'id': test_ids
})

for label in labels:
    print(f"Processing label: {label}")
    print(subtables[label].head())
    print(subtables[label].shape)
    original_smiles = subtables[label]['SMILES'].tolist()
    original_labels = subtables[label][label].values
    original_smiles, original_labels = augment_smiles_dataset(original_smiles, original_labels, num_augments=1)
    fingerprints, descriptors, valid_smiles, invalid_indices\
    =smiles_to_combined_fingerprints_with_descriptors(original_smiles, radius=2, n_bits=128)
    descriptors = [descriptors[i] for i in range(len(descriptors)) if i not in invalid_indices]
    fingerprints = np.delete(fingerprints, invalid_indices, axis=0)
    X=pd.DataFrame(descriptors)
    X=X.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge', 'SMILES'],axis=1, errors='ignore')
    y = np.delete(original_labels, invalid_indices)
    X = X.filter(filters[label])
    fp_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])
    print(fp_df.shape)
    fp_df.reset_index(drop=True, inplace=True)
    X.reset_index(drop=True, inplace=True)
    X = pd.concat([X, fp_df], axis=1)
    print(f"After concat: {X.shape}")
    threshold = 0.01
    selector = VarianceThreshold(threshold=threshold)
    X = selector.fit_transform(X)
    print(f"After variance cut: {X.shape}")
    n_samples = 1000
    X, y = augment_dataset(X, y, n_samples=n_samples)
    print(f"After augment cut: {X.shape}")
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
    if label=="Tg":
        Model= XGBRegressor(n_estimators= 2173, learning_rate= 0.0672418745539774, max_depth= 6, reg_lambda= 5.545520219149715)
    if label=='Rg':
        Model = XGBRegressor(n_estimators= 520, learning_rate= 0.07324113948440986, max_depth= 5, reg_lambda=0.9717380315982088)
    if label=='FFV':
        Model = XGBRegressor(n_estimators= 2202, learning_rate= 0.07220580588586338, max_depth= 4, reg_lambda= 2.8872976032666493)
    if label=='Tc':
        Model = XGBRegressor(n_estimators= 1488, learning_rate= 0.010456188013762864, max_depth= 5, reg_lambda= 9.970345982204618)
    if label=='Density':
        Model = XGBRegressor(n_estimators= 1958, learning_rate= 0.10955287548172478, max_depth= 5, reg_lambda= 3.074470087965767)
    Model.fit(X_train,y_train)
    y_pred=Model.predict(X_test)
    
    print(mean_absolute_error(y_pred,y_test))
    Model.fit(X,y)
    fingerprints, descriptors, valid_smiles, invalid_indices\
    =smiles_to_combined_fingerprints_with_descriptors(test_smiles, radius=2, n_bits=128)
    descriptors = [descriptors[i] for i in range(len(descriptors)) if i not in invalid_indices]
    fingerprints = np.delete(fingerprints, invalid_indices, axis=0)
    test=pd.DataFrame(descriptors)
    test=test.drop(['BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge', 'SMILES'],axis=1, errors='ignore')
    test = test.filter(filters[label])
    fp_df = pd.DataFrame(fingerprints, columns=[f'FP_{i}' for i in range(fingerprints.shape[1])])
    fp_df.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    test = pd.concat([test, fp_df], axis=1)
    test = selector.transform(test)
    print(test.shape)
    y_pred=Model.predict(test).flatten()
    print(y_pred)
    new_column_name = label
    output_df[new_column_name] = y_pred
    print(output_df)
    
output_df.to_csv('submission.csv', index=False)




