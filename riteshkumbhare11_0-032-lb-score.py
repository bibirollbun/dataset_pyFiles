import kagglehub
path = kagglehub.dataset_download("senkin13/rdkit-2025-3-3-cp311")
print("Path to dataset files", path)


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


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



tg=pd.read_csv('/kaggle/input/neurips-dataset/tg.csv')
rg=pd.read_csv('/kaggle/input/neurips-dataset/rg.csv')
tc=pd.read_csv('/kaggle/input/neurips-dataset/tc.csv')
ffv=pd.read_csv('/kaggle/input/neurips-dataset/ffv.csv')
density=pd.read_csv('/kaggle/input/neurips-dataset/density.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ID=test['id'].copy()


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan
test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))


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

test = pd.concat([test, preprocessing(test)], axis=1)
test['Ipc']=np.log10(test['Ipc'])

test=test.drop(['id','SMILES'],axis=1)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def model_seed_all(train_d, test_d, model, target, seed_id=1, submission=False):
    X = train_d.drop(columns=[target])
    y = train_d[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

    Model = model(random_state=seed_id)

    if not submission:
        Model.fit(X_train, y_train)
        y_pred = Model.predict(X_val)
        return mean_absolute_error(y_val, y_pred)
    else:
        Model.fit(X, y)
        return Model.predict(test_d)


import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

def average_predictions(train_df, test_df, target_col):
    seeds = [i * 100 for i in range(1, 5)]  
    preds = [
        model_seed_all(train_df, test_df, ExtraTreesRegressor, target_col, seed_id=seed, submission=True)
        for seed in seeds
    ]
    return np.mean(preds, axis=0)


tg_result      = average_predictions(tg,      test, 'Tg')
ffv_result     = average_predictions(ffv,     test, 'FFV')
tc_result      = average_predictions(tc,      test, 'Tc')
density_result = average_predictions(density, test, 'Density')
rg_result      = average_predictions(rg,      test, 'Rg')


 # Finally, we use the model to predict on the test set and prepare the submission file.

sub={'id':ID,'Tg':tg_result,
     'FFV':ffv_result,
     'Tc':tc_result,
     'Density':density_result,
     'Rg':rg_result}


submission=pd.DataFrame(sub)


sub=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv', dtype=str)


sub['Tg']=tg_result
sub['FFV']=ffv_result
sub['Tc']=tc_result
sub['Density']=density_result
sub['Rg']=rg_result


submission


sub.to_csv('submission.csv',index=False)




