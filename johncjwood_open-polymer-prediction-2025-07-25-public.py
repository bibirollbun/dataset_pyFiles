## README
## First, navigate to the required datasets in another tab
# https://www.kaggle.com/datasets/senkin13/rdkit-2025-3-3-cp311
# https://www.kaggle.com/datasets/minatoyukinaxlisa/tc-smiles
# https://www.kaggle.com/datasets/dmitryuarov/smiles-extra-data
# When submitting to Kaggle
# Upload from Colab
## In Kaggle:
# Under input, add 3 data sets and the competition
# Input -> Competition Filter -> NeurIPS
# Input -> tc-smiles
# Input -> rdkit-2025-3-3-cp311
# Input -> smiles-extra-data
# Settings -> Turn off internet
# Check that isGoogleDrive variable is False
# Run All




## Recognition
# https://www.kaggle.com/code/samithsachidanandan/neurips-rdkit-multi-models-lb-0-033
# https://www.kaggle.com/code/dmitryuarov/neurips-baseline-external-data/notebook
# for pulling in external data, useless columns, and rdkit
# https://www.kaggle.com/code/alejandrolopezrincon/extra-data-with-fs-starting-point
# cleaning the test data



#set to True for Google Colab
#set to False for Kaggle
#isGoogleDrive = True
isGoogleDrive = False


#set to the second option for Kaggle
if isGoogleDrive:
  !pip install rdkit catboost
else:
  !pip install --quiet /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

import networkx as nx
from rdkit.Chem import AllChem, Draw, Descriptors, rdmolops
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

#For Pytorch
import torch
import copy
import tqdm
import numpy as np
import matplotlib.pyplot as plt

#using gpu for training
torch.cuda.is_available()
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_default_device()
print(device)

class CFG:
  TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
  SEED = 42
  FOLDS = 5


#Set to True for Kaggle competitition notebook update
#dataframes
default_path = '/kaggle/input/neurips-open-polymer-prediction-2025/'
train_path = default_path + 'train.csv'
test_path = default_path + 'test.csv'
tc_smiles_path = '/kaggle/input/tc-smiles/Tc_SMILES.csv'
tg_smiles_path = '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv'
ktg_smiles_path = '/kaggle/input/smiles-extra-data/data_tg3.xlsx'
de_smiles_path = '/kaggle/input/smiles-extra-data/data_dnst1.xlsx'
dataset1_Tc_path = default_path + 'train_supplement/dataset1.csv'
dataset3_Tg_path = default_path + 'train_supplement/dataset3.csv'
dataset4_FFV_path = default_path + 'train_supplement/dataset4.csv'
#ss_path = default_path + 'sample_submission.csv'
df_path = 'df.csv'
submission_path = 'submission.csv'
pickle_path = 'df.pkl'
best_weights_path = 'best_weights.pth'


if isGoogleDrive:
  from google.colab import drive
  drive.mount('/content/drive')
  train_path = '/content/drive/My Drive/Kaggle2025/kaggle/train.csv'
  test_path = '/content/drive/My Drive/Kaggle2025/kaggle/test.csv'
  tc_smiles_path = '/content/drive/My Drive/Kaggle2025/kaggle/Tc_SMILES.csv'
  tg_smiles_path = '/content/drive/My Drive/Kaggle2025/kaggle/JCIM_sup_bigsmiles.csv'
  ktg_smiles_path = '/content/drive/My Drive/Kaggle2025/kaggle/data_tg3.xlsx'
  de_smiles_path = '/content/drive/My Drive/Kaggle2025/kaggle/data_dnst1.xlsx'
  dataset1_Tc_path = '/content/drive/My Drive/Kaggle2025/kaggle/train_supplement/dataset1.csv'
  dataset3_Tg_path = '/content/drive/My Drive/Kaggle2025/kaggle/train_supplement/dataset3.csv'
  dataset4_FFV_path = '/content/drive/My Drive/Kaggle2025/kaggle/train_supplement/dataset4.csv'
  #ss_path = '/content/drive/My Drive/Kaggle2025/kaggle/sample_submission.csv'
  df_path = '/content/drive/My Drive/Kaggle2025/df.csv'
  submission_path = '/content/drive/My Drive/Kaggle2025/submission.csv'
  pickle_path = '/content/drive/My Drive/Kaggle2025/df.pkl'
  best_weights_path = '/content/drive/My Drive/Kaggle2025/best_weights.pth'


#We will load both the training and test datasets using pandas, and store test IDs
train = pd.read_csv(train_path)
train['src'] = 'train'
#print("train len:" + str(len(train)))
test_orig = pd.read_csv(test_path, dtype=str)
test = test_orig.copy()
test['src'] = 'test'
#print("test len:" + str(len(test)))
#ss =pd.read_csv(ss_path)

#Train Suppliment
dataset1_Tc = pd.read_csv(dataset1_Tc_path)
dataset1_Tc = dataset1_Tc.rename(columns={'TC_mean': 'Tc'})
dataset1_Tc['src'] = 'dataset1_Tc'

dataset3_Tg = pd.read_csv(dataset3_Tg_path)
dataset3_Tg['src'] = 'dataset3_Tg'

dataset4_FFV = pd.read_csv(dataset4_FFV_path)
dataset4_FFV['src'] = 'dataset4_FFV'

#smiles tc data -> https://www.kaggle.com/datasets/minatoyukinaxlisa/tc-smiles
#smiles extra data -> https://www.kaggle.com/datasets/dmitryuarov/smiles-extra-data
tc_smiles = pd.read_csv(tc_smiles_path)
tc_smiles.rename(columns={'TC_mean': 'Tc'}, inplace=True)
tc_smiles['src'] = 'tc_smiles'
#print("tc_smiles len:" + str(len(tc_smiles)))

tg_smiles =pd.read_csv(tg_smiles_path)
tg_smiles.rename(columns={'Tg (C)': 'Tg'}, inplace=True)
tg_smiles['src'] = 'tg_smiles'
#print("tg_smiles len:" + str(len(tg_smiles)))

ktg_smiles =pd.read_excel(ktg_smiles_path)
ktg_smiles.rename(columns={'Tg [K]': 'Tg'}, inplace=True)
ktg_smiles['Tg'] = ktg_smiles['Tg'] - 273.15
ktg_smiles['src'] = 'ktg_smiles'
#print("ktg_smiles len:" + str(len(ktg_smiles)))

de_smiles =pd.read_excel(de_smiles_path)
de_smiles.rename(columns={'density(g/cm3)': 'Density'}, inplace=True)
de_smiles = de_smiles[(de_smiles['SMILES'].notnull())&(de_smiles['Density'].notnull())&(de_smiles['Density'] != 'nylon')]
de_smiles['Density'] = de_smiles['Density'].astype('float64')
de_smiles['Density'] -= 0.118
de_smiles['src'] = 'de_smiles'
#print("de_smiles len:" + str(len(de_smiles)))

## Unify smile format
def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
  """Completely clean and validate SMILES, removing all problematic patterns"""
  if not isinstance(smile, str) or len(smile) == 0:
      return None
  # List of all problematic patterns we've seen
  bad_patterns = [
      '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]',
      "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
      # Additional patterns that cause issues
      '([R])', '([R1])', '([R2])',
  ]
  # Check for any bad patterns
  for pattern in bad_patterns:
      if pattern in smile:
          return np.nan
  # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
  if '][' in smile and any(x in smile for x in ['[R', 'R]']):
      return np.nan
  try:
      mol = Chem.MolFromSmiles(smile)
      canon_smile = Chem.MolToSmiles(mol, canonical=True)
      return canon_smile
  except:
      return np.nan

train['SMILES_can'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
test['SMILES_can'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))
tc_smiles['SMILES_can'] = tc_smiles['SMILES'].apply(lambda s: make_smile_canonical(s))
tg_smiles['SMILES_can'] = tg_smiles['SMILES'].apply(lambda s: make_smile_canonical(s))
ktg_smiles['SMILES_can'] = ktg_smiles['SMILES'].apply(lambda s: make_smile_canonical(s))
de_smiles['SMILES_can'] = de_smiles['SMILES'].apply(lambda s: make_smile_canonical(s))
dataset1_Tc['SMILES_can'] = dataset1_Tc['SMILES'].apply(lambda s: make_smile_canonical(s))
dataset3_Tg['SMILES_can'] = dataset3_Tg['SMILES'].apply(lambda s: make_smile_canonical(s))
dataset4_FFV['SMILES_can'] = dataset4_FFV['SMILES'].apply(lambda s: make_smile_canonical(s))

train = train[train['SMILES_can'].notnull()].reset_index(drop=True)
test = test[test['SMILES_can'].notnull()].reset_index(drop=True)

tc_smiles = tc_smiles[tc_smiles['SMILES_can'].notnull()].reset_index(drop=True) #'Tc'
tc_smiles = tc_smiles.groupby(['SMILES_can','src'])['Tc'].mean()
tc_smiles = tc_smiles.reset_index()

tg_smiles = tg_smiles[tg_smiles['SMILES_can'].notnull()].reset_index(drop=True) #'Tg'
tg_smiles = tg_smiles.groupby(['SMILES_can','src'])['Tg'].mean()
tg_smiles = tg_smiles.reset_index()

ktg_smiles = ktg_smiles[ktg_smiles['SMILES_can'].notnull()].reset_index(drop=True) #'Tg'
ktg_smiles = ktg_smiles.groupby(['SMILES_can','src'])['Tg'].mean()
ktg_smiles = ktg_smiles.reset_index()

de_smiles = de_smiles[de_smiles['SMILES_can'].notnull()].reset_index(drop=True) #'Density'
de_smiles = de_smiles.groupby(['SMILES_can','src'])['Density'].mean()
de_smiles = de_smiles.reset_index()

dataset1_Tc = dataset1_Tc[dataset1_Tc['SMILES_can'].notnull()].reset_index(drop=True) #'Tc'
dataset1_Tc = dataset1_Tc.groupby(['SMILES_can','src'])['Tc'].mean()
dataset1_Tc = dataset1_Tc.reset_index()

dataset3_Tg = dataset3_Tg[dataset3_Tg['SMILES_can'].notnull()].reset_index(drop=True) #'Tg'
dataset3_Tg = dataset3_Tg.groupby(['SMILES_can','src'])['Tg'].mean()
dataset3_Tg = dataset3_Tg.reset_index()

dataset4_FFV = dataset4_FFV[dataset4_FFV['SMILES_can'].notnull()].reset_index(drop=True) #'FFV'
dataset4_FFV = dataset4_FFV.groupby(['SMILES_can','src'])['FFV'].mean()
dataset4_FFV = dataset4_FFV.reset_index()

## Combining datasets
# want to take all this data and combine it into one big dataset for the data engineering exercise
# Add missing columns to each dataframe with NaN values to align them for concatenation
total_columns = list(set(train.columns) | set(tc_smiles.columns)
  | set(test.columns) | set(tg_smiles.columns) | set(ktg_smiles.columns) | set(de_smiles.columns)
  )
all_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg','src','id','SMILES','SMILES_can','uSMILES','BigSMILES','std_name','tradenames','synonyms']
def add_missing_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df[columns] # Ensure column order is consistent

train_aligned = add_missing_columns(train.copy(), all_columns)
test_aligned = add_missing_columns(test.copy(), all_columns)
tg_smiles_aligned = add_missing_columns(tg_smiles.copy(), all_columns)
tc_smiles_aligned = add_missing_columns(tc_smiles.copy(), all_columns)
ktg_smiles_aligned = add_missing_columns(ktg_smiles.copy(), all_columns)
de_smiles_aligned = add_missing_columns(de_smiles.copy(), all_columns)
dataset1_Tc_aligned = add_missing_columns(dataset1_Tc.copy(), all_columns)
dataset3_Tg_aligned = add_missing_columns(dataset3_Tg.copy(), all_columns)
dataset4_FFV_aligned = add_missing_columns(dataset4_FFV.copy(), all_columns)


# Concatenate the dataframes
def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    #this group by is causing the loss of columns - need to expand the columns
    #df_extra = df_extra.groupby('SMILES_can', as_index=False)[target].mean()
    df_extra.drop_duplicates(subset=['SMILES_can'],keep='first', inplace=True)
    cross_smiles = set(df_extra['SMILES_can']) & set(df_train['SMILES_can'])
    unique_smiles_extra = set(df_extra['SMILES_can']) - set(df_train['SMILES_can'])

    # Make priority target value from competition's df
    for smile in df_train[df_train[target].notnull()]['SMILES_can'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)

    # Imput missing values for competition's SMILES
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES_can']==smile, target] = df_extra[df_extra['SMILES_can']==smile][target].values[0]

    df_train = pd.concat([df_train, df_extra[df_extra['SMILES_can'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    #print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    #print(f'New unique SMILES: {len(unique_smiles_extra)}')
    return df_train

df = pd.concat([train_aligned, test_aligned], ignore_index=True)
df = add_extra_data(df, tc_smiles_aligned, 'Tc')
df = add_extra_data(df, tg_smiles_aligned, 'Tg')
df = add_extra_data(df, ktg_smiles_aligned, 'Tg')
df = add_extra_data(df, de_smiles_aligned, 'Density')
df = add_extra_data(df, dataset1_Tc_aligned, 'Tc')
df = add_extra_data(df, dataset3_Tg_aligned, 'Tg')
df = add_extra_data(df, dataset4_FFV_aligned, 'FFV')
#drop invalid rows
#print("pre-df remove null smiles_can len:" + str(len(df)))
df = df[df['SMILES_can'].notnull()].reset_index(drop=True)
#print("post-df remove null smiles_can len:" + str(len(df)))

#display(df.head())



# confirm there are no more duplicates
duplicate_smiles = df[df.duplicated(subset=['SMILES_can'], keep=False)]
display(duplicate_smiles.sort_values(by='SMILES_can'))


useless_cols = [

    'MaxPartialCharge',
    # Nan data
    'BCUT2D_MWHI',
    'BCUT2D_MWLOW',
    'BCUT2D_CHGHI',
    'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI',
    'BCUT2D_LOGPLOW',
    'BCUT2D_MRHI',
    'BCUT2D_MRLOW',

    # Constant data
    'NumRadicalElectrons',
    'SMR_VSA8',
    'SlogP_VSA9',
    'fr_barbitur',
    'fr_benzodiazepine',
    'fr_dihydropyridine',
    'fr_epoxide',
    'fr_isothiocyan',
    'fr_lactam',
    'fr_nitroso',
    'fr_prisulfonamd',
    'fr_thiocyan',

    # High correlated data >0.95
    'MaxEStateIndex',
    'HeavyAtomMolWt',
    'ExactMolWt',
    'NumValenceElectrons',
    'Chi0',
    'Chi0n',
    'Chi0v',
    'Chi1',
    'Chi1n',
    'Chi1v',
    'Chi2n',
    'Kappa1',
    'LabuteASA',
    'HeavyAtomCount',
    'MolMR',
    'Chi3n',
    'BertzCT',
    'Chi2v',
    'Chi4n',
    'HallKierAlpha',
    'Chi3v',
    'Chi4v',
    'MinAbsPartialCharge',
    'MinPartialCharge',
    'MaxAbsPartialCharge',
    'FpDensityMorgan2',
    'FpDensityMorgan3',
    'Phi',
    'Kappa3',
    'fr_nitrile',
    'SlogP_VSA6',
    'NumAromaticCarbocycles',
    'NumAromaticRings',
    'fr_benzene',
    'VSA_EState6',
    'NOCount',
    'fr_C_O',
    'fr_C_O_noCOO',
    'NumHDonors',
    'fr_amide',
    'fr_Nhpyrrole',
    'fr_phenol',
    'fr_phenol_noOrthoHbond',
    'fr_COO2',
    'fr_halogen',
    'fr_diazo',
    'fr_nitro_arom',
    'fr_phos_ester'
]

def compute_all_descriptors(smiles,desc_names):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]

def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))

def preprocessing(df):
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    descriptors = [compute_all_descriptors(smi,desc_names) for smi in df['SMILES_can'].to_list()]

    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    for smile in df['SMILES_can']:
         compute_graph_features(smile, graph_feats)

    result = pd.concat(
        [
            pd.DataFrame(descriptors, columns=desc_names),
            pd.DataFrame(graph_feats)
        ],axis=1)

    result = result.replace([-np.inf, np.inf], np.nan)
    return result
df = pd.concat([df, preprocessing(df)], axis=1)


# Find constant columns for each target
# First 14 columns are not to be touched
all_features = df.columns[13:].tolist()
features = {}
for target in CFG.TARGETS:
    const_descs = []
    for col in df.columns.drop(CFG.TARGETS):
        if df[df[target].notnull()][col].nunique() == 1:
            const_descs.append(col)
    features[target] = [f for f in all_features if f not in const_descs]
#do we even want to do this?
#take the natural log of IPC
df['Ipc']=np.log(df['Ipc'])

#this is probably not the best practice to replace the values with the mean
for n in train.columns[13:]:
    df[n]=df[n].replace(-np.inf,np.nan)
    df[n]=df[n].replace(np.inf,np.nan)
    df[n].fillna(df[n].median())

#print(df.shape)


## Preparing data for model phase ##
#separate the train and test data
train = df[df['src']!='test'].reset_index(drop=True)
test = df[df['src']=='test'].reset_index(drop=True)
ID=test['id'].copy()

# We'll separate train to be one model for each target variable.
t_1=train[['SMILES_can','Tg']].copy()
t_2=train[['SMILES_can','FFV']].copy()
t_3=train[['SMILES_can','Tc']].copy()
t_4=train[['SMILES_can','Density']].copy()
t_5=train[['SMILES_can','Rg']].copy()

# We will drop the rows with missing values related to that target after separation.
#This is important , dropping them beforehand would result Null for all data.
t_1.dropna(inplace=True)
t_2.dropna(inplace=True)
t_3.dropna(inplace=True)
t_4.dropna(inplace=True)
t_5.dropna(inplace=True)

#drop non-numeric fields at the top of the dataset
train = train.drop(['Tg','FFV','Tc','Density','Rg','id','src','SMILES','uSMILES','BigSMILES','std_name','tradenames','synonyms'],axis=1)
test = test.drop(['Tg','FFV','Tc','Density','Rg','id','src','SMILES','uSMILES','BigSMILES','std_name','tradenames','synonyms','SMILES_can'],axis=1)

tg=t_1.merge(train,on='SMILES_can',how='left')
ffv=t_2.merge(train,on='SMILES_can',how='left')
tc=t_3.merge(train,on='SMILES_can',how='left')
density=t_4.merge(train,on='SMILES_can',how='left')
rg=t_5.merge(train,on='SMILES_can',how='left')

for i in (tg,tc,density,ffv,rg):
    i.drop('SMILES_can',axis=1,inplace=True)
    i.replace([np.inf, -np.inf], np.nan, inplace=True)
    i.dropna(inplace=True)
    i = i.astype(np.float32)


from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def train_and_evaluate(train_d, test_d, model_class, target, submission=False, model_params=None):
    X = train_d.drop(target, axis=1)
    y = train_d[target].copy()

    if model_params is None:
        model_params = {}
    model = model_class(**model_params)

    if not submission:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)
    else:
        model.fit(X, y)
        return model.predict(test_d)


sub = {
    'id': ID,
    'Tg': 1.0 * train_and_evaluate(tg, test, ExtraTreesRegressor, 'Tg', submission=True)

    ,'FFV': 0.5 * train_and_evaluate(ffv, test, ExtraTreesRegressor, 'FFV', submission=True)
           + 0.5 * train_and_evaluate(ffv, test, XGBRegressor, 'FFV', submission=True)

    ,'Tc': 0.5 * train_and_evaluate(tc, test, ExtraTreesRegressor, 'Tc', submission=True)
          + 0.5 * train_and_evaluate(tc, test, XGBRegressor, 'Tc', submission=True)

    ,'Density': train_and_evaluate(density, test, ExtraTreesRegressor, 'Density', submission=True)

    ,'Rg': train_and_evaluate(rg, test, ExtraTreesRegressor, 'Rg', submission=True),
}
submission_subset=pd.DataFrame(sub)




#take all the original ids and left join in the submission_subset by the id column
submission = test_orig.merge(submission_subset,on='id',how='left')
submission = submission.rename(columns={'SMILES': 'SMILES_can'})
submission = submission.set_index('SMILES_can')

#overwrite any cases where the smile exists in the original data
tc_smiles_indexed = tc_smiles.set_index('SMILES_can') #'Tc'
tg_smiles_indexed = tg_smiles.set_index('SMILES_can') #'Tg'
ktg_smiles_indexed = ktg_smiles.set_index('SMILES_can') #'Tg'
de_smiles_indexed = de_smiles.set_index('SMILES_can') #'Density'
dataset1_Tc_indexed = dataset1_Tc.set_index('SMILES_can') #'Tc'
dataset3_Tg_indexed = dataset3_Tg.set_index('SMILES_can') #'Tg'
dataset4_FFV_indexed = dataset4_FFV.set_index('SMILES_can') #'FFV'
#update using the indexes
submission.update(tc_smiles_indexed[['Tc']])
submission.update(tg_smiles_indexed[['Tg']])
submission.update(ktg_smiles_indexed[['Tg']])
submission.update(de_smiles_indexed[['Density']])
submission.update(dataset1_Tc_indexed[['Tc']])
submission.update(dataset3_Tg_indexed[['Tg']])
submission.update(dataset4_FFV_indexed[['FFV']])

#drop the smiles column and submission index
submission = submission.reset_index()
submission = submission.drop(['SMILES_can'],axis=1)

#For each dataframe, calculate the median of a particular column
tg_mean = tg['Tg'].median()
ffv_mean = ffv['FFV'].median()
tc_mean = tc['Tc'].median()
density_mean = density['Density'].median()
rg_mean = rg['Rg'].median()
#fill in all the N/A values in the submission with these mean values
submission = submission.replace(-np.inf,np.nan)
submission = submission.replace(np.inf,np.nan)
submission['Tg'].fillna(tg_mean, inplace=True)
submission['FFV'].fillna(ffv_mean, inplace=True)
submission['Tc'].fillna(tc_mean, inplace=True)
submission['Density'].fillna(density_mean, inplace=True)
submission['Rg'].fillna(rg_mean, inplace=True)

submission.to_csv('submission.csv', index=False)
if isGoogleDrive == False:
  submission.to_csv('/kaggle/working/submission.csv', index=False)


submission.head()

