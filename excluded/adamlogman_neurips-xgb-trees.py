!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


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


import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
# Show all columns
from rdkit import Chem
from mordred import Calculator, descriptors
pd.set_option('display.max_columns', None)

import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops


tg_=pd.read_csv('/kaggle/input/new-dataset/tg.csv')
rg_=pd.read_csv('/kaggle/input/new-dataset/rg.csv')
tc_=pd.read_csv('/kaggle/input/new-dataset/tc.csv')
ffv_=pd.read_csv('/kaggle/input/new-dataset/ffv.csv')
density_=pd.read_csv('/kaggle/input/new-dataset/density.csv')
test_=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


tg=pd.read_csv('/kaggle/input/modred-dataset/desc_tg.csv')
tc=pd.read_csv('/kaggle/input/modred-dataset/desc_tc.csv')
rg=pd.read_csv('/kaggle/input/modred-dataset/desc_rg.csv')
ffv=pd.read_csv('/kaggle/input/modred-dataset/desc_ffv.csv')
density=pd.read_csv('/kaggle/input/modred-dataset/desc_de.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ID=test['id']


for i in (tg,tc,rg,ffv,density):
     i.drop(columns=[col for col in i.columns if i[col].nunique() == 1],axis=1,inplace=True)


# Remove columns with object or category dtype
tg = tg.select_dtypes(exclude=['object', 'category'])
rg = rg.select_dtypes(exclude=['object', 'category'])
ffv = ffv.select_dtypes(exclude=['object', 'category'])
tc = tc.select_dtypes(exclude=['object', 'category'])
density  = density.select_dtypes(exclude=['object', 'category'])


tg.shape


tc.shape


density.shape


rg.shape


mols_test = [Chem.MolFromSmiles(s) for s in test.SMILES]

# Initialize the Mordred Calculator
calc = Calculator(descriptors, ignore_3D=True) # ignore_3D=True for 2D descriptors

desc_test = calc.pandas(mols_test)


# desc_test.drop(columns=[col for col in desc_test.columns if desc_test[col].nunique() == 1],axis=1,inplace=True)
# desc_test = desc_test.select_dtypes(exclude=['object', 'category'])


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan
test_['SMILES'] = test_['SMILES'].apply(lambda s: make_smile_canonical(s))


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


def compute_all_descriptors(smiles):
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

test_ = pd.concat([test_, preprocessing(test_)], axis=1)
test_['Ipc']=np.log10(test_['Ipc'])

test_=test_.drop(['id','SMILES'],axis=1)





def model(train_d,test_d,model,target,submission=False):
    # We divide the data into training and validation sets for model evaluation
    train_cols = set(train_d.columns) - {target}
    test_cols = set(test_d.columns)
   # Intersect the feature columns
    common_cols = list(train_cols & test_cols)
    X=train_d[common_cols].copy()
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model
    if submission==False:
       Model.fit(X_train,y_train)
       y_pred=Model.predict(X_test)
       return mean_absolute_error(y_pred,y_test)         # We assess our model performance using MAE metric
    if submission==True:
       Model.fit(X,y)
       submission=Model.predict(test_d[common_cols].copy())
       return submission


# model(tc,,'Tc')
# model(tg,desc_test,CatBoostRegressor(verbose=False),'Tg',submission=False)


# model(tc,,'Tc')
# model(tg_,test_,CatBoostRegressor(verbose=False),'Tg',submission=False)


# model(ffv,desc_test,LGBMRegressor,'FFV',submission=False)


# model(tc,desc_test,LGBMRegressor,'Tc',submission=False)


# model(density,desc_test,LGBMRegressor,'Density',submission=False)


# model(rg,desc_test,LGBMRegressor,'Rg',submission=False)


# tg.shape


# desc_test.shape


# model(tg,desc_test,LGBMRegressor,'Tg',submission=True)


 # Finally, we use the model to predict on the test set and prepare the submission file.

sub={'id':ID,'Tg':model(tg_,test_,CatBoostRegressor(verbose=False),'Tg',submission=True),
     'FFV':model(ffv_,test_,CatBoostRegressor(verbose=False),'FFV',submission=True),
     'Tc':model(tc_,test_,CatBoostRegressor(verbose=False),'Tc',submission=True),
     'Density':model(density_,test_,CatBoostRegressor(verbose=False),'Density',submission=True),
     'Rg':model(rg_,test_,CatBoostRegressor(verbose=False),'Rg',submission=True)}


submission=pd.DataFrame(sub)


submission


submission.to_csv('submission.csv',index=False)

