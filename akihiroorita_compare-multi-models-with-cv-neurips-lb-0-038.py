!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


 # Importing Required Libraries\nLet's begin by importing the essential Python libraries needed for data processing, visualization, and modeling.

import pandas as pd
import numpy as np


from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from catboost import CatBoostRegressor
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


class CFG:
    TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    SEED = 42
    FOLDS = 5


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



 #We will load both the training and test datasets using pandas, and store test IDs 
train=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ss=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
ID=test['id'].copy()


tc_smiles =pd.read_csv('/kaggle/input/tc-smiles/Tc_SMILES.csv')
tg_smiles =pd.read_csv('/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv')
ktg_smiles =pd.read_excel('/kaggle/input/smiles-extra-data/data_tg3.xlsx')
de_smiles =pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
ffv_smiles =pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv')


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

train['SMILES'] = train['SMILES'].apply(lambda s: make_smile_canonical(s))
test['SMILES'] = test['SMILES'].apply(lambda s: make_smile_canonical(s))


ktg_smiles.rename(columns={'Tg [K]': 'Tg'}, inplace=True)
tg_smiles.rename(columns={'Tg (C)': 'Tg'}, inplace=True)
tc_smiles.rename(columns={'TC_mean': 'Tc'}, inplace=True)
de_smiles.rename(columns={'density(g/cm3)': 'Density'}, inplace=True)
ffv_smiles.rename(columns={'FFV': 'FFV'}, inplace=True)


de_smiles['SMILES'] = de_smiles['SMILES'].apply(lambda s: make_smile_canonical(s))
de_smiles = de_smiles[(de_smiles['SMILES'].notnull())&(de_smiles['Density'].notnull())&(de_smiles['Density'] != 'nylon')]
de_smiles['Density'] = de_smiles['Density'].astype('float64')
de_smiles['Density'] -= 0.118

ktg_smiles['Tg'] = ktg_smiles['Tg'] - 273.15


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


train



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

train = add_extra_data(train, tc_smiles, 'Tc')
train = add_extra_data(train, tg_smiles, 'Tg')
train = add_extra_data(train, ktg_smiles, 'Tg')
train = add_extra_data(train, de_smiles, 'Density')
train = add_extra_data(train, ffv_smiles, 'FFV')


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

train = pd.concat([train, preprocessing(train)], axis=1)
test = pd.concat([test, preprocessing(test)], axis=1)

# Find constant columns for each target
all_features = train.columns[7:].tolist()
features = {}
for target in CFG.TARGETS:
    const_descs = []
    for col in train.columns.drop(CFG.TARGETS):
        if train[train[target].notnull()][col].nunique() == 1:
            const_descs.append(col)
    features[target] = [f for f in all_features if f not in const_descs]

print(train.shape)
train['Ipc']=np.log10(train['Ipc'])  
for n in train.columns[7:]:
    train[n]=train[n].replace(-np.inf,np.nan)
    train[n]=train[n].replace(np.inf,np.nan)    
    train[n].fillna(train[n].mean())
  
print(train.shape)
test['Ipc']=np.log10(test['Ipc'])
for n in test.columns[7:]:
    train[n]=train[n].replace(-np.inf,np.nan)
    train[n]=train[n].replace(np.inf,np.nan)      
    test[n].fillna(train[n].mean())


# We'll separate train to be one model for each target variable.
t_1=train[['SMILES','Tg']].copy()
t_2=train[['SMILES','FFV']].copy()
t_3=train[['SMILES','Tc']].copy()
t_4=train[['SMILES','Density']].copy()
t_5=train[['SMILES','Rg']].copy()

# We will drop the rows with missing values related to that target after separation.
#This is important , dropping them beforehand would result Null for all data.
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


train


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np

def model_cv(train_d, model_fn, target, n_splits=5):
    X = train_d.drop(columns=[target])
    y = train_d[target].copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = model_fn()
        m.fit(X_train, y_train)
        y_pred = m.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        maes.append(mae)
        print(f"Fold {fold}: MAE = {mae:.5f}")

    return np.mean(maes), np.std(maes)


from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor


models_to_compare = {
    'RandomForest': lambda: RandomForestRegressor(),
    'ExtraTrees': lambda: ExtraTreesRegressor(),
    'GradientBoosting': lambda: GradientBoostingRegressor(),
    'HistGradientBoosting': lambda: HistGradientBoostingRegressor(),
    'XGBoost': lambda: XGBRegressor(verbosity=0),
    'LightGBM': lambda: LGBMRegressor(verbose=-1),
    'CatBoost': lambda: CatBoostRegressor(verbose=0),
    'Ridge': lambda: Ridge(),
    'Lasso': lambda: Lasso(),
    'ElasticNet': lambda: ElasticNet(),
    'Huber': lambda: HuberRegressor(),
    'KNN': lambda: KNeighborsRegressor(),
    'SVR': lambda: SVR(),
    'AdaBoost': lambda: AdaBoostRegressor(),
}

for name, model_cls in models_to_compare.items():
    try:
        mean_mae, std_mae = model_cv(tg, model_cls, 'Tg', n_splits=5)
        print(f'{name:<20}: MAE = {mean_mae:.5f} Â± {std_mae:.5f}')
    except Exception as e:
        print(f'{name:<20}: Failed with error: {e}')


for name, model_cls in models_to_compare.items():
    try:
        mean_mae, std_mae = model_cv(ffv, model_cls, 'FFV', n_splits=5)
        print(f'{name:<20}: MAE = {mean_mae:.5f} Â± {std_mae:.5f}')
    except Exception as e:
        print(f'{name:<20}: Failed with error: {e}')











models_to_compare = {
    'RandomForest': lambda: RandomForestRegressor(),
    'ExtraTrees': lambda: ExtraTreesRegressor(),
    'GradientBoosting': lambda: GradientBoostingRegressor(),
    'HistGradientBoosting': lambda: HistGradientBoostingRegressor(),
    'XGBoost': lambda: XGBRegressor(verbosity=0),
    'LightGBM': lambda: LGBMRegressor(verbose=-1),
    'CatBoost': lambda: CatBoostRegressor(verbose=0),
    'Ridge': lambda: Ridge(),
    'Lasso': lambda: Lasso(),
    'ElasticNet': lambda: ElasticNet(),
    'Huber': lambda: HuberRegressor(),
    'KNN': lambda: KNeighborsRegressor(),
    'SVR': lambda: SVR(),
    'AdaBoost': lambda: AdaBoostRegressor(),
}

for name, model_cls in models_to_compare.items():
    try:
        mean_mae, std_mae = model_cv(tc, model_cls, 'Tc', n_splits=5)
        print(f'{name:<20}: MAE = {mean_mae:.5f} Â± {std_mae:.5f}')
    except Exception as e:
        print(f'{name:<20}: Failed with error: {e}')





models_to_compare = {
    'RandomForest': lambda: RandomForestRegressor(),
    'ExtraTrees': lambda: ExtraTreesRegressor(),
    'GradientBoosting': lambda: GradientBoostingRegressor(),
    'HistGradientBoosting': lambda: HistGradientBoostingRegressor(),
    'XGBoost': lambda: XGBRegressor(verbosity=0),
    'LightGBM': lambda: LGBMRegressor(verbose=-1),
    'CatBoost': lambda: CatBoostRegressor(verbose=0),
    'Ridge': lambda: Ridge(),
    'Lasso': lambda: Lasso(),
    'ElasticNet': lambda: ElasticNet(),
    'Huber': lambda: HuberRegressor(),
    'KNN': lambda: KNeighborsRegressor(),
    'SVR': lambda: SVR(),
    'AdaBoost': lambda: AdaBoostRegressor(),
}

for name, model_cls in models_to_compare.items():
    try:
        mean_mae, std_mae = model_cv(density, model_cls, 'Density', n_splits=5)
        print(f'{name:<20}: MAE = {mean_mae:.5f} Â± {std_mae:.5f}')
    except Exception as e:
        print(f'{name:<20}: Failed with error: {e}')


models_to_compare = {
    'RandomForest': lambda: RandomForestRegressor(),
    'ExtraTrees': lambda: ExtraTreesRegressor(),
    'GradientBoosting': lambda: GradientBoostingRegressor(),
    'HistGradientBoosting': lambda: HistGradientBoostingRegressor(),
    'XGBoost': lambda: XGBRegressor(verbosity=0),
    'LightGBM': lambda: LGBMRegressor(verbose=-1),
    'CatBoost': lambda: CatBoostRegressor(verbose=0),
    'Ridge': lambda: Ridge(),
    'Lasso': lambda: Lasso(),
    'ElasticNet': lambda: ElasticNet(),
    'Huber': lambda: HuberRegressor(),
    'KNN': lambda: KNeighborsRegressor(),
    'SVR': lambda: SVR(),
    'AdaBoost': lambda: AdaBoostRegressor(),
}

for name, model_cls in models_to_compare.items():
    try:
        mean_mae, std_mae = model_cv(rg, model_cls, 'Rg', n_splits=5)
        print(f'{name:<20}: MAE = {mean_mae:.5f} Â± {std_mae:.5f}')
    except Exception as e:
        print(f'{name:<20}: Failed with error: {e}')


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from catboost import CatBoostRegressor

# ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ä»˜ã��ãƒ¢ãƒ‡ãƒ«äºˆæ¸¬é–¢æ•°
def model_cv_predict(train_d, test_d, model_fn, target, n_splits=5):
    X = train_d.drop(columns=[target])
    y = train_d[target].copy()
    test_preds = np.zeros(len(test_d))
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = model_fn()
        m.fit(X_train, y_train)

        val_pred = m.predict(X_val)
        mae = mean_absolute_error(y_val, val_pred)
        maes.append(mae)

        test_preds += m.predict(test_d) / n_splits
        print(f"Fold {fold}: MAE = {mae:.5f}")

    print(f"CV MAE = {np.mean(maes):.5f} Â± {np.std(maes):.5f}")
    return test_preds

# submission ç”¨äºˆæ¸¬
sub = {
    'id': ID,
    'Tg': model_cv_predict(tg, test, lambda: CatBoostRegressor(verbose=0), 'Tg'),
    'FFV': model_cv_predict(ffv, test, lambda: ExtraTreesRegressor(), 'FFV'),
    'Tc': model_cv_predict(tc, test, lambda: CatBoostRegressor(verbose=0), 'Tc'),
    'Density': model_cv_predict(density, test, lambda: ExtraTreesRegressor(), 'Density'),
    'Rg': model_cv_predict(rg, test, lambda: ExtraTreesRegressor(), 'Rg')
}

# CSVã�¨ã�—ã�¦ä¿�å­˜
submission_df = pd.DataFrame(sub)
submission_df.to_csv("submission.csv", index=False)
print("âœ… submission.csv ã‚’ä¿�å­˜ã�—ã�¾ã�—ã�Ÿ")


submission_df




