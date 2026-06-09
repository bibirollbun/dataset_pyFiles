# installing rdkit
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


# importing required libraries
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import networkx as nx
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)



# reading datasets
train = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
test_ids=test['id'].copy()

dataset1 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv')
dataset3 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv')
dataset4 = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')

dataset5 = pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
dataset6 = pd.read_csv('/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv')
dataset7 = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
dataset8 = pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
dataset9 = pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv')


# columns not needed
useless_cols = [
    'MaxPartialCharge', 
    # Nan data
    'BCUT2D_MWHI','BCUT2D_MWLOW','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRHI','BCUT2D_MRLOW',
    # Constant data
    'NumRadicalElectrons','SMR_VSA8','SlogP_VSA9','fr_barbitur','fr_benzodiazepine','fr_dihydropyridine','fr_epoxide','fr_isothiocyan','fr_lactam','fr_nitroso','fr_prisulfonamd','fr_thiocyan',
    # High correlated data >0.95
    'MaxEStateIndex','HeavyAtomMolWt','ExactMolWt','NumValenceElectrons','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Kappa1','LabuteASA','HeavyAtomCount','MolMR','Chi3n','BertzCT','Chi2v','Chi4n','HallKierAlpha','Chi3v','Chi4v','MinAbsPartialCharge','MinPartialCharge','MaxAbsPartialCharge','FpDensityMorgan2','FpDensityMorgan3','Phi','Kappa3','fr_nitrile','SlogP_VSA6','NumAromaticCarbocycles','NumAromaticRings','fr_benzene','VSA_EState6','NOCount','fr_C_O','fr_C_O_noCOO','NumHDonors','fr_amide','fr_Nhpyrrole','fr_phenol','fr_phenol_noOrthoHbond','fr_COO2','fr_halogen','fr_diazo','fr_nitro_arom','fr_phos_ester'
]



# makes smiles canonical using Chem library
def make_smile_canonical(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))



# find all descriptors derived from the smile
def compute_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [None] * len(desc_names)
    return [desc[1](mol) for desc in Descriptors.descList if desc[0] not in useless_cols]



# add more data to the datasets
def add_extra_data(df_train, df_extra, target):
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(lambda s: make_smile_canonical(s))
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Make priority target value from competition's df
    for smile in df_train[df_train[target].notnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            cross_smiles.remove(smile)

    # Imput missing values for competition's SMILES
    for smile in cross_smiles:
        df_train.loc[df_train['SMILES']==smile, target] = df_extra[df_extra['SMILES']==smile][target].values[0]
    
    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], axis=0).reset_index(drop=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'\nFor target "{target}" added {n_samples_after-n_samples_before} new samples!')
    print(f'New unique SMILES: {len(unique_smiles_extra)}')
    return df_train

dataset1.rename(columns={'TC_mean': 'Tc'}, inplace=True)
train = add_extra_data(train, dataset3, 'Tg')
train = add_extra_data(train, dataset4, 'FFV')
train = add_extra_data(train, dataset1, 'Tc')

# dataset5.rename(columns={'TC_mean': 'Tc'}, inplace=True)
# dataset9.rename(columns={'Tg (C)': 'Tg'}, inplace=True)
# dataset8.rename(columns={'Tg [K]': 'Tg'}, inplace=True)
dataset7 = dataset7.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']].query('SMILES.notnull() and Density.notnull() and Density != "nylon"').assign(Density=lambda x: x['Density'].astype(float) - 0.118)

# train = add_extra_data(train, dataset6, 'Tg')
# train = add_extra_data(train, dataset9, 'Tg')
# train = add_extra_data(train, dataset5, 'Tc')
train = add_extra_data(train, dataset7, 'Density')
# train = add_extra_data(train, dataset8, 'Tg')


# compute graphical features from the smile
def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))



# pre-process data - add extra fields, graphical fields, remove useless columns, etc
def preprocessing(df):
    desc_names = [desc[0] for desc in Descriptors.descList if desc[0] not in useless_cols]
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES'].to_list()]

    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': []}
    for smile in df['SMILES']:
         compute_graph_features(smile, graph_feats)
        
    result = pd.concat(
        [
            pd.DataFrame(descriptors, columns=desc_names),
            pd.DataFrame(graph_feats)
        ],
        axis=1
    )

    result = result.replace([-np.inf, np.inf], np.nan)
    return result

train = pd.concat([train, preprocessing(train)], axis=1)
test = pd.concat([test, preprocessing(test)], axis=1)



TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# find constant columns for each target
all_features = train.columns[7:].tolist()
features = {}
for target in TARGETS:
    const_descs = []
    for col in train.columns.drop(TARGETS):
        subset = train.loc[train[target].notnull(), col]        
        nunique_result = subset.nunique()
        
        if isinstance(nunique_result, pd.Series):
            # Multiple columns selected
            if (nunique_result == 1).all():
                const_descs.append(col)
        else:
            # Single column selected
            if nunique_result == 1:
                const_descs.append(col)
    features[target] = [f for f in all_features if f not in const_descs]

train['Ipc']=np.log10(train['Ipc'])  
for n in train.columns[7:]:
    train[n]=train[n].replace(-np.inf,np.nan)
    train[n]=train[n].replace(np.inf,np.nan)    
    train[n].fillna(train[n].mean())
  
test['Ipc']=np.log10(test['Ipc'])
for n in test.columns[7:]:
    train[n]=train[n].replace(-np.inf,np.nan)
    train[n]=train[n].replace(np.inf,np.nan)      
    test[n].fillna(train[n].mean())



# creating separate dataset for each target
t_1=train[['SMILES','Tg']].copy()
t_2=train[['SMILES','FFV']].copy()
t_3=train[['SMILES','Tc']].copy()
t_4=train[['SMILES','Density']].copy()
t_5=train[['SMILES','Rg']].copy()

t_1.dropna(inplace=True)
t_2.dropna(inplace=True)
t_3.dropna(inplace=True)
t_4.dropna(inplace=True)
t_5.dropna(inplace=True)

train=train.drop(['id','Tg','FFV','Tc','Density','Rg'],axis=1)
test=test.drop(['id','SMILES'],axis=1)

tg=t_1.merge(train,on='SMILES',how='left')
ffv=t_2.merge(train,on='SMILES',how='left')
tc=t_3.merge(train,on='SMILES',how='left')
density=t_4.merge(train,on='SMILES',how='left')
rg=t_5.merge(train,on='SMILES',how='left')

for i in (tg,tc,density,ffv,rg):
    i.drop('SMILES',axis=1,inplace=True)
    i.dropna(inplace=True)



# remove duplicates from datasets
tg = tg.loc[:, ~tg.columns.duplicated(keep="first")]
ffv = ffv.loc[:, ~ffv.columns.duplicated(keep="first")]
tc = tc.loc[:, ~tc.columns.duplicated(keep="first")]
density = density.loc[:, ~density.columns.duplicated(keep="first")]
rg = rg.loc[:, ~rg.columns.duplicated(keep="first")]



# train, test using model
def model(train_d,test_d,target):
    X=train_d.drop(target,axis=1)
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=CatBoostRegressor()
    Model.fit(X,y)
    submission=Model.predict(test_d)
    return submission


# finding submission results
sub={'id':test_ids,
     'Tg':model(tg,test,'Tg'),
     'FFV':model(ffv,test,'FFV'),
     'Tc':model(tc,test,'Tc'),
     'Density':model(density,test,'Density'),
     'Rg':model(rg,test,'Rg')}


# create submission.csv
submission=pd.DataFrame(sub)
submission.to_csv('submission.csv',index=False)
submission

