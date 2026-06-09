!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import polars as pl
import torch
import gc
import pickle

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import warnings
import os
import gc
import time
import seaborn as sns
from importlib.metadata import version
import matplotlib.pyplot as plt
print("matplotlib version:", version("matplotlib"))
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem
import networkx as nx
from rdkit.Chem import Descriptors, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import rdBase
from rdkit.Chem import rdmolops
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings('ignore')


import transformers
from transformers import RobertaTokenizer, TFRobertaModel, RobertaModel, RobertaForMaskedLM
transformers.logging.set_verbosity_error()
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import sys
import os
import yaml
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
pd.set_option("display.max_columns", None)


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 500)


data_path = r"/kaggle/input/private-dataset-new"
path = r"/kaggle/input/neurips-open-polymer-prediction-2025"


train_tg1 = pd.read_csv(os.path.join(data_path, "df_tg1.csv"))
train_tc = pd.read_csv(os.path.join(data_path, "df_tc1.csv"))
train_tg2 = pd.read_csv(os.path.join(data_path, "df_tg2.csv"))
train_ffv = pd.read_csv(os.path.join(data_path, "df_ffv1.csv"))
train_density = pd.read_csv(os.path.join(data_path, "df_density1.csv"))
train_rg = pd.read_csv(os.path.join(data_path, "df_rg1.csv"))


test1 = pd.read_csv(os.path.join(path, "test.csv"))


useless_cols1 = ['NumRadicalElectrons', "MaxPartialCharge", 'MinAbsPartialCharge',
                'MinPartialCharge','MaxAbsPartialCharge','FpDensityMorgan2',
                'FpDensityMorgan3','BCUT2D_MWHI', 'BCUT2D_MWLOW',
                'BCUT2D_CHGHI', 'BCUT2D_CHGLO','BCUT2D_LOGPHI',
                'BCUT2D_LOGPLOW','BCUT2D_MRHI','BCUT2D_MRLOW',
                'BCUT2D_MWHI', 'BCUT2D_MWLOW','BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI','BCUT2D_LOGPLOW',
 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'Chi0v','Chi1v', 'Chi2v' , 'Chi3v', 'Chi4v','FpDensityMorgan2',
                'FpDensityMorgan3',
                 'Chi4v','SMR_VSA8','SlogP_VSA9',
                 'NumSpiroAtoms','fr_Al_COO', 'fr_Ar_COO',
                 'fr_Ar_NH', 'fr_COO', 'fr_COO2',
                 'fr_HOCCN', 'fr_N_O', 'fr_Ndealkylation1',
                 'fr_Nhpyrrole','fr_SH','fr_aldehyde',
                 'fr_azide', 'fr_barbitur', 'fr_benzodiazepine',
                 'fr_diazo', 'fr_dihydropyridine','fr_epoxide',
                 'fr_guanido', 'fr_hdrzone', 'fr_isocyan',          
                 'fr_isothiocyan','fr_lactam','fr_morpholine',
                 'fr_nitroso', 'fr_phos_acid','fr_phos_ester',                 
                 'fr_piperdine','fr_piperzine', 'fr_priamide',
                 'fr_prisulfonamd', 'fr_quatN',   'fr_sulfonamd',
                 'fr_term_acetylene', 'fr_tetrazole',   'fr_thiocyan',
                 'mfp_0', 'mfp_6',  'mfp_12',  'mfp_16',
                 'mfp_17', 'mfp_18',  'mfp_20', 'mfp_22',
                 'mfp_23', 'mfp_30', 'mfp_35', 'mfp_38',
                 'mfp_40','mfp_43','mfp_44', 'mfp_46','mfp_47',
                 'mfp_51', 'mfp_54', 'mfp_55', 'mfp_56', 'mfp_61',
                 'mfp_68', 'mfp_75',  'mfp_76', 'mfp_77',
                 'mfp_78', 'mfp_81', 'mfp_82',  'mfp_83', 'mfp_85',
                 'mfp_86', 'mfp_88', 'mfp_89','mfp_91', 'mfp_92',
                 'mfp_95', 'mfp_96', 'mfp_98', 'mfp_99',
                 'mfp_101', 'mfp_103','mfp_104','mfp_106',
                 'mfp_107',  'mfp_108','mfp_109','mfp_110',
                 'mfp_111','mfp_113', 'mfp_115',  'mfp_120',
                 'mfp_122',  'mfp_126','mfp_127','mfp_129',
                 'mfp_130','mfp_131', 'mfp_132', 'mfp_133',
                 'mfp_134','mfp_141', 'mfp_142',  'mfp_143',
                 'mfp_146','mfp_148','mfp_149','mfp_153','mfp_154',
                 'mfp_155','mfp_156','mfp_159','mfp_163','mfp_164',
                 'mfp_165','mfp_166','mfp_168','mfp_169','mfp_173',
                 'mfp_174','mfp_177','mfp_178','mfp_181','mfp_182',
                 'mfp_183','mfp_185','mfp_186','mfp_187','mfp_189',
                 'mfp_195','mfp_196','mfp_198','mfp_199', 'mfp_201',
                 'mfp_206','mfp_207','mfp_208','mfp_209','mfp_211','mfp_213',
                 'mfp_215','mfp_217','mfp_218',
                 'mfp_221',
                 'mfp_223',
                 'mfp_229',
                 'mfp_230',
                 'mfp_234',
                 'mfp_236',
                 'mfp_238',
                 'mfp_242',
                 'mfp_246',
                 'mfp_248',
                 'mfp_254',
                 'mfp_256',
                 'mfp_257',
                 'mfp_263',
                 'mfp_265',
                 'mfp_269',
                 'mfp_271',
                 'mfp_272',
                 'mfp_273',
                 'mfp_276',
                 'mfp_277',
                 'mfp_278',
                 'mfp_281',
                 'mfp_282',
                 'mfp_286',
                 'mfp_291',
                 'mfp_295',
                 'mfp_296',
                 'mfp_298',
                 'mfp_299',
                 'mfp_303',
                 'mfp_307',
                 'mfp_309',
                 'mfp_312',
                 'mfp_313',
                 'mfp_316',
                 'mfp_317',
                 'mfp_318',
                 'mfp_320',
                 'mfp_321',
                 'mfp_324',
                 'mfp_331',
                 'mfp_334',
                 'mfp_335',
                 'mfp_336',
                 'mfp_337',
                 'mfp_338',
                 'mfp_339',
                 'mfp_343',
                 'mfp_344',
                 'mfp_345',
                 'mfp_347',
                 'mfp_348',
                 'mfp_349',
                 'mfp_351',
                 'mfp_353',
                 'mfp_354',
                 'mfp_355',
                 'mfp_357',
                 'mfp_363',
                 'mfp_364',
                 'mfp_365',
                 'mfp_368',
                 'mfp_369',
                 'mfp_370',
                 'mfp_371',
                 'mfp_372',
                 'mfp_373',
                 'mfp_376',
                 'mfp_377',
                 'mfp_379',
                 'mfp_380',
                 'mfp_382',
                 'mfp_384',
                 'mfp_385',
                 'mfp_388',
                 'mfp_390',
                 'mfp_394',
                 'mfp_395',
                 'mfp_397',
                 'mfp_398',
                 'mfp_399',
                 'mfp_400',
                 'mfp_402',
                 'mfp_403',
                 'mfp_404',
                 'mfp_405',
                 'mfp_407',
                 'mfp_408',
                 'mfp_409',
                 'mfp_410',
                 'mfp_414',
                 'mfp_415',
                 'mfp_418',
                 'mfp_424',
                 'mfp_425',
                 'mfp_426',
                 'mfp_427',
                 'mfp_431',
                 'mfp_435',
                 'mfp_437',
                 'mfp_438',
                 'mfp_439',
                 'mfp_441',
                 'mfp_442',
                 'mfp_443',
                 'mfp_444',
                 'mfp_445',
                 'mfp_446',
                 'mfp_447',
                 'mfp_448',
                 'mfp_450',
                 'mfp_451',
                 'mfp_453',
                 'mfp_454',
                 'mfp_455',
                 'mfp_459',
                 'mfp_460',
                 'mfp_462',
                 'mfp_464',
                 'mfp_465',
                 'mfp_466',
                 'mfp_467',
                 'mfp_468',
                 'mfp_470',
                 'mfp_472',
                 'mfp_474',
                 'mfp_475',
                 'mfp_476',
                 'mfp_477',
                 'mfp_478',
                 'mfp_480',
                 'mfp_481',
                 'mfp_488',
                 'mfp_489',
                 'mfp_490',
                 'mfp_491',
                 'mfp_493',
                 'mfp_495',
                 'mfp_496',
                 'mfp_497',
                 'mfp_499',
                 'mfp_500',
                 'mfp_502',
                 'mfp_505',
                 'mfp_506',
                 'mfp_507',
                 'mfp_508',
                 'mfp_509',
                 'mfp_513',
                 'mfp_514',
                 'mfp_515',
                 'mfp_516',
                 'mfp_517',
                 'mfp_523',
                 'mfp_524',
                 'mfp_525',
                 'mfp_527',
                 'mfp_528',
                 'mfp_529',
                 'mfp_530',
                 'mfp_531',
                 'mfp_532',
                 'mfp_533',
                 'mfp_534',
                 'mfp_535',
                 'mfp_536',
                 'mfp_542',
                 'mfp_543',
                 'mfp_544',
                 'mfp_546',
                 'mfp_550',
                 'mfp_551',
                 'mfp_553',
                 'mfp_554',
                 'mfp_560',
                 'mfp_563',
                 'mfp_564',
                 'mfp_566',
                 'mfp_567',
                 'mfp_570',
                 'mfp_571',
                 'mfp_572',
                 'mfp_574',
                 'mfp_575',
                 'mfp_576',
                 'mfp_577',
                 'mfp_579',
                 'mfp_582',
                 'mfp_583',
                 'mfp_589',
                 'mfp_590',
                 'mfp_595',
                 'mfp_600',
                 'mfp_603',
                 'mfp_604',
                 'mfp_605',
                 'mfp_608',
                 'mfp_611',
                 'mfp_612',
                 'mfp_613',
                 'mfp_615',
                 'mfp_616',
                 'mfp_618',
                 'mfp_619',
                 'mfp_627',
                 'mfp_628',
                 'mfp_631',
                 'mfp_633',
                 'mfp_634',
                 'mfp_635',
                 'mfp_636',
                 'mfp_637',
                 'mfp_638',
                 'mfp_640',
                 'mfp_643',
                 'mfp_646',
                 'mfp_648',
                 'mfp_649',
                 'mfp_651',
                 'mfp_652',
                 'mfp_653',
                 'mfp_654',
                 'mfp_655',
                 'mfp_660',
                 'mfp_661',
                 'mfp_665',
                 'mfp_666',
                 'mfp_668',
                 'mfp_671',
                 'mfp_678',
                 'mfp_681',
                 'mfp_685',
                 'mfp_687',
                 'mfp_688',
                 'mfp_690',
                 'mfp_693',
                 'mfp_696',
                 'mfp_697',
                 'mfp_702',
                 'mfp_704',
                 'mfp_706',
                 'mfp_707',
                 'mfp_708',
                 'mfp_711',
                 'mfp_712',
                 'mfp_713',
                 'mfp_716',
                 'mfp_717',
                 'mfp_719',
                 'mfp_720',
                 'mfp_724',
                 'mfp_727',
                 'mfp_729',
                 'mfp_732',
                 'mfp_733',
                 'mfp_735',
                 'mfp_737',
                 'mfp_738',
                 'mfp_740',
                 'mfp_744',
                 'mfp_748',
                 'mfp_750',
                 'mfp_752',
                 'mfp_754',
                 'mfp_755',
                 'mfp_756',
                 'mfp_757',
                 'mfp_758',
                 'mfp_759',
                 'mfp_760',
                 'mfp_762',
                 'mfp_763',
                 'mfp_764',
                 'mfp_765',
                 'mfp_767',
                 'mfp_768',
                 'mfp_770',
                 'mfp_772',
                 'mfp_774',
                 'mfp_776',
                 'mfp_778',
                 'mfp_779',
                 'mfp_780',
                 'mfp_782',
                 'mfp_783',
                 'mfp_784',
                 'mfp_787',
                 'mfp_788',
                 'mfp_789',
                 'mfp_791',
                 'mfp_793',
                 'mfp_796',
                 'mfp_797',
                 'mfp_798',
                 'mfp_800',
                 'mfp_806',
                 'mfp_808',
                 'mfp_809',
                 'mfp_810',
                 'mfp_812',
                 'mfp_813',
                 'mfp_814',
                 'mfp_817',
                 'mfp_818',
                 'mfp_820',
                 'mfp_821',
                 'mfp_824',
                 'mfp_825',
                 'mfp_826',
                 'mfp_837',
                 'mfp_839',
                 'mfp_840',
                 'mfp_844',
                 'mfp_845',
                 'mfp_846',
                 'mfp_847',
                 'mfp_850',
                 'mfp_851',
                 'mfp_853',
                 'mfp_855',
                 'mfp_858',
                 'mfp_860',
                 'mfp_861',
                 'mfp_865',
                 'mfp_867',
                 'mfp_868',
                 'mfp_869',
                 'mfp_870',
                 'mfp_871',
                 'mfp_873',
                 'mfp_874',
                 'mfp_876',
                 'mfp_877',
                 'mfp_882',
                 'mfp_883',
                 'mfp_884',
                 'mfp_885',
                 'mfp_889',
                 'mfp_892',
                 'mfp_894',
                 'mfp_897',
                 'mfp_899',
                 'mfp_903',
                 'mfp_906',
                 'mfp_907',
                 'mfp_908',
                 'mfp_909',
                 'mfp_911',
                 'mfp_912',
                 'mfp_913',
                 'mfp_914',
                 'mfp_915',
                 'mfp_917',
                 'mfp_919',
                 'mfp_920',
                 'mfp_921',
                 'mfp_922',
                 'mfp_923',
                 'mfp_925',
                 'mfp_928',
                 'mfp_933',
                 'mfp_934',
                 'mfp_937',
                 'mfp_941',
                 'mfp_943',
                 'mfp_948',
                 'mfp_949',
                 'mfp_950',
                 'mfp_952',
                 'mfp_955',
                 'mfp_957',
                 'mfp_959',
                 'mfp_963',
                 'mfp_970',
                 'mfp_977',
                 'mfp_979',
                 'mfp_981',
                 'mfp_983',
                 'mfp_985',
                 'mfp_986',
                 'mfp_987',
                 'mfp_988',
                 'mfp_989',
                 'mfp_990',
                 'mfp_991',
                 'mfp_992',
                 'mfp_994',
                 'mfp_995',
                 'mfp_996',
                 'mfp_1000',
                 'mfp_1001',
                 'mfp_1002',
                 'mfp_1005',
                 'mfp_1006',
                 'mfp_1007',
                 'mfp_1010',
                 'mfp_1015',
                 'mfp_1016',
                 'mfp_1020',
                 'mfp_1021',
                 'mfp_1022',
                 'mfp_1023',
             ]


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan


# --- Feature Engineering ---
def compute_all_descriptors(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    desc_list = [desc[0] for desc in Descriptors._descList if desc[0] not in useless_cols1]
    morgan_fp_size = 1024
    if mol is None: return np.full(len(desc_list) + morgan_fp_size, np.nan)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)
    descriptors = np.array(calculator.CalcDescriptors(mol))
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=morgan_fp_size)
    mfp_array = np.array(list(mfp.ToBitString())).astype(int)
    return np.concatenate([descriptors, mfp_array])


def compute_graph_features(smiles, graph_feats):
    mol = Chem.MolFromSmiles(smiles)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)

    graph_feats['graph_diameter'].append(nx.diameter(G) if nx.is_connected(G) else 0)
    graph_feats['avg_shortest_path'].append(nx.average_shortest_path_length(G) if nx.is_connected(G) else 0)
    graph_feats['num_cycles'].append(len(list(nx.cycle_basis(G))))

    # Extended features
    graph_feats['graph_density'].append(nx.density(G))
    graph_feats['avg_degree'].append(np.mean([d for _, d in G.degree()]))
    graph_feats['avg_clustering'].append(nx.average_clustering(G))
    graph_feats['assortativity'].append(nx.degree_assortativity_coefficient(G))

    # Optional centralities
    betweenness = nx.betweenness_centrality(G)
    graph_feats['avg_betweenness'].append(np.mean(list(betweenness.values())))

    try:
        eigen = nx.eigenvector_centrality_numpy(G)
        graph_feats['avg_eigenvector'].append(np.mean(list(eigen.values())))
    except:
        graph_feats['avg_eigenvector'].append(np.nan)  # fail-saf


def preprocessing(df):
    df['SMILES'] = df['SMILES'].apply(lambda s: make_smile_canonical(s))
    df['SM_len'] = df['SMILES'].map(lambda x: len(x))
    df['SM_len'] = df['SM_len'].map(lambda x: int(x))
    desc_names = [desc[0] for desc in Descriptors._descList if desc[0] not in useless_cols1]
#     desc_names = [d[0] for d in Descriptors._descList]
    descriptors = [compute_all_descriptors(smi) for smi in df['SMILES'].to_list()]
    fp_morgan_cols = [f'mfp_{i}' for i in range(1024)]
    fp_morgan_cols1 = [x for x in fp_morgan_cols if x in useless_cols1]
    feature_columns = desc_names + fp_morgan_cols

    graph_feats = {'graph_diameter': [], 'avg_shortest_path': [], 'num_cycles': [], 'graph_density': [],
                  'avg_degree': [], 'avg_clustering': [], 'assortativity': [], 
                  'avg_betweenness': [], 'avg_eigenvector': []}
    
    for smile in df['SMILES']:
         compute_graph_features(smile, graph_feats)
        
    result = pd.concat(
        [
            pd.DataFrame(descriptors, columns=feature_columns),
            pd.DataFrame(graph_feats),
        ],
        axis=1
    )
    result = result.drop(fp_morgan_cols1, axis=1)
    result = result.replace([-np.inf, np.inf], np.nan)
    return result


df_test1 = pd.concat([test1, preprocessing(test1)], axis=1)


df_test1


display(train_tc.shape)
display(train_ffv.shape)
display(train_density.shape)
display(train_rg.shape)
display(train_tg1.shape)
display(train_tg2.shape)



train_tc.head()


import optuna
# --------------------------- CONFIG ---------------------------
class CFG:
    # General
    TARGET_COLS = ["Tg", "FFV", "Tc", "Density", "Rg"]
    N_FOLDS = 5
    SEED = 42
    MODELS_TO_RUN = ["lgbm", "xgb", "catboost", ]

    # Feature Engineering & Selection
    N_FEATURES_TO_SELECT = 686 # Select top N features for each target

    # Hyperparameter Tuning
    USE_HYPERTUNING = False  # Set to False to use default params
    N_TRIALS = 5  # Number of optimization trials per model
    TUNING_TIMEOUT = 100  # INCREASE THIS: Timeout in seconds (recommend 1 hour)

    # Default Model Parameters (used if hypertuning is disabled)


    LGBM_PARAMS = {
        "objective": "mae",
        "metric": "mae",
        "n_estimators": 3000,
        "learning_rate": 0.05,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 0.05,
        "num_leaves": 128,
        "verbose": -1,
        "n_jobs": -1,
        "seed": SEED,
        "boosting_type": "gbdt",
    }

    XGB_PARAMS = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_jobs": -1,
        "verbose": 0,
        "tree_method": "hist",
        "seed": SEED,
    }

    CATBOOST_PARAMS = {
        "objective": "MAE",
        "eval_metric": "MAE",
        "iterations": 1500,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 0.1,
        "random_strength": 0.1,
        "loss_function": "MAE",
        "random_seed": SEED,
        "verbose": 0,
        "thread_count": -1,
    }

# -------------------------- UTILITY FUNCTIONS --------------------------

def create_stratified_folds(df, target_col, n_splits, seed):
    """Create stratified folds for regression by binning the target."""
    df['bins'] = pd.cut(df[target_col], bins=10, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(X=df, y=df['bins']))
#     folds = list(skf.split(X=df, y=df['bins']))
    df = df.drop('bins', axis=1)
    return folds

# def select_features(train_df, test_df, features, target_col, n_features):
def select_features(train_df, features, target_col, n_features):
    """Select top features using LightGBM feature importance."""
    print(f"  Performing feature selection for {target_col}...")
    temp_model = lgb.LGBMRegressor(**CFG.LGBM_PARAMS)
    temp_model.fit(train_df[features], train_df[target_col])
    importances = pd.DataFrame({
        'feature': features,
        'importance': temp_model.feature_importances_
    }).sort_values('importance', ascending=False)

    top_features = importances['feature'].head(n_features).tolist()
    print(f"  Selected {len(top_features)} features.")
    return top_features

# -------------------------- HYPERPARAMETER TUNING --------------------------

def suggest_lgbm_params(trial):
    """Suggest hyperparameters for LightGBM."""
    return {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0, log=True),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "verbose": -1,
        "n_jobs": -1,
        "seed": CFG.SEED,
    }


def suggest_xgb_params(trial):
    """Suggest hyperparameters for XGBoost."""
    return {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_jobs": -1,
        "verbose": 0,
        "tree_method": "hist",
        "seed": CFG.SEED,
    }

def suggest_catboost_params(trial):
    """Suggest hyperparameters for CatBoost."""
    return {
        "objective": "MAE",
        "eval_metric": "MAE",
        "loss_function": "MAE",
        "iterations": 1500,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_seed": CFG.SEED,
        "verbose": 0,
        "thread_count": -1,
    }



def objective_function(trial, model_name, X_train, y_train, folds):
    """Objective function for hyperparameter optimization."""

    # Get model parameters based on model type
    if model_name == 'lgbm':
        params = suggest_lgbm_params(trial)
        model_class = lgb.LGBMRegressor
    elif model_name == 'xgb':
        params = suggest_xgb_params(trial)
        model_class = xgb.XGBRegressor
    elif model_name == 'catboost':
        params = suggest_catboost_params(trial)
        model_class = cat.CatBoostRegressor

    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Create model
        model = model_class(**params)

        # Create pipeline
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5)),
            # ('scaler', StandardScaler()),
            ('model', model)
        ])

        # Prepare validation data for early stopping
        imputer = KNNImputer(n_neighbors=5)
        scaler = StandardScaler()
        X_tr_processed = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_val_processed = scaler.transform(imputer.transform(X_val))
        

        # Set up fit parameters

        if model_name == 'lgbm':
            fit_params = {
                'model__eval_set': [(X_val_processed, y_val)],
                'model__callbacks': [lgb.early_stopping(100, verbose=False)]
            }
        elif model_name == 'xgb':
            fit_params = {
                'model__eval_set': [(X_val_processed, y_val)],
                'model__callbacks': [xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                'model__verbose': False
            }
        elif model_name == 'catboost':
            fit_params = {
                'model__eval_set': [(X_val_processed, y_val)],
                'model__early_stopping_rounds': 100
            }

        # Fit and predict
        pipeline.fit(X_tr, y_tr, **fit_params)
        pred = pipeline.predict(X_val)
        score = mean_absolute_error(y_val, pred)
        cv_scores.append(score)

    return np.mean(cv_scores)

def tune_hyperparameters(model_name, X_train, y_train, folds, n_trials=500, timeout=100):
    """Tune hyperparameters using Optuna."""
    print(f"    Tuning hyperparameters for {model_name.upper()}...")

    study = optuna.create_study(direction='minimize', 
                               sampler=optuna.samplers.TPESampler(seed=CFG.SEED))

    study.optimize(
        lambda trial: objective_function(trial, model_name, X_train, y_train, folds),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    print(f"    Best {model_name.upper()} score: {study.best_value:.6f}")
    print(f"    Best {model_name.upper()} params: {study.best_params}")

    return study.best_params


processed_test_df = df_test1.copy()


features = train_ffv.columns[2:]
len(features)


# -------------------------- MODELING PIPELINE --------------------------

print("Starting enhanced model training with hyperparameter tuning...\n")
start_time = time.time()

test_predictions = pd.DataFrame({"id": processed_test_df["id"]})
oof_predictions = {}
best_params_per_target = {}

# initial_features = [
#     col for col in train_data.columns
#     if col not in ["id", "SMILES", "weight", "Unnamed: 0"] + CFG.TARGET_COLS and col in processed_test_df.columns
# ]

for target in CFG.TARGET_COLS:
    print(f"--- Training models for {target} ---")

    # Prepare data for the current target
#     train_df = processed_train_df.dropna(subset=[target]).reset_index(drop=True)
    if target == "Tg":
        train_df = train_tg2.copy()
    elif target == "Tc":
        train_df = train_tc.copy()
    elif target == "FFV":
        train_df = train_ffv.copy()
    elif target == "Density":
        train_df = train_density.copy()
    elif target == "Rg":
        train_df = train_rg.copy()
#     train_df = sample_data.copy()
    y_train = train_df[target]

    # Dynamic Feature Selection
    selected_features = select_features(train_df, features, target, CFG.N_FEATURES_TO_SELECT)
    
    X_train = train_df[selected_features]
    X_test = processed_test_df[selected_features]

    # Create stratified folds for this target
    folds = create_stratified_folds(train_df, target, CFG.N_FOLDS, CFG.SEED)

    # Store best parameters for this target
    best_params_per_target[target] = {}

    # Hyperparameter tuning phase
    if CFG.USE_HYPERTUNING:
        print(f"\n  === HYPERPARAMETER TUNING PHASE ===")
        for model_name in CFG.MODELS_TO_RUN:
            best_params = tune_hyperparameters(
                model_name, X_train, y_train, folds, 
                CFG.N_TRIALS, CFG.TUNING_TIMEOUT
            )
            best_params_per_target[target][model_name] = best_params

    # Training phase with best parameters
    print(f"\n  === TRAINING PHASE ===")
    oof_preds_ensemble = np.zeros((len(X_train), len(CFG.MODELS_TO_RUN)))
    test_preds_ensemble = np.zeros((len(X_test), len(CFG.MODELS_TO_RUN)))

    for model_idx, model_name in enumerate(CFG.MODELS_TO_RUN):
        print(f"\n  Training {model_name.upper()} model...")

        # Get parameters (tuned or default)
        if CFG.USE_HYPERTUNING:
            if model_name == 'lgbm':
                model_params = best_params_per_target[target][model_name]

            elif model_name == 'xgb':
                model_params = best_params_per_target[target][model_name]

            elif model_name == 'catboost':
                model_params = best_params_per_target[target][model_name]

        else:

            if model_name == 'lgbm':
                model_params = CFG.LGBM_PARAMS
            elif model_name == 'xgb':
                model_params = CFG.XGB_PARAMS
            elif model_name == 'catboost':
                model_params = CFG.CATBOOST_PARAMS

        for fold, (train_idx, val_idx) in enumerate(folds):
            print(f"    Fold {fold + 1}/{CFG.N_FOLDS}")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Create model with best parameters
   
            if model_name == 'lgbm':
                model = lgb.LGBMRegressor(**model_params)
            elif model_name == 'xgb':
                model = xgb.XGBRegressor(**model_params)
            elif model_name == 'catboost':
                model = cat.CatBoostRegressor(**model_params)

            # Create pipeline
            pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler()),
                ('model', model)
            ])

            # Prepare validation data for early stopping
            imputer = KNNImputer(n_neighbors=5)
            scaler = StandardScaler()
            X_tr_processed = scaler.fit_transform(imputer.fit_transform(X_tr))
            X_val_processed = scaler.transform(imputer.transform(X_val))

            # Set up fit parameters
    
            if model_name == 'lgbm':
                fit_params = {
                    'model__eval_set': [(X_val_processed, y_val)],
                    'model__callbacks': [lgb.early_stopping(100, verbose=False)]
                }
            elif model_name == 'xgb':
                fit_params = {
                    'model__eval_set': [(X_val_processed, y_val)],
                    'model__callbacks': [xgb.callback.EarlyStopping(rounds=100, save_best=True)],
                    'model__verbose': False
                }
            elif model_name == 'catboost':
                fit_params = {
                    'model__eval_set': [(X_val_processed, y_val)],
                    'model__early_stopping_rounds': 100,
                }

            # Fit and predict
            pipeline.fit(X_tr, y_tr, **fit_params)
            oof_preds_ensemble[val_idx, model_idx] = pipeline.predict(X_val)
            test_preds_ensemble[:, model_idx] += pipeline.predict(X_test) / CFG.N_FOLDS

    # Average the predictions from all models (simple ensemble)
    final_oof = np.mean(oof_preds_ensemble, axis=1)
    final_preds = np.mean(test_preds_ensemble, axis=1)

    test_predictions[target] = final_preds
    oof_predictions[target] = final_oof
    print(f"\nEnsemble MAE for {target}: {mean_absolute_error(y_train, final_oof):.4f}\n")
    gc.collect()


# -------------------------- SUBMISSION & SUMMARY --------------------------


submission_df = test_predictions.copy()
# if "Tg" in submission_df.columns:
#     submission_df["Tg"] += 273.15 # Postprocess Tg to Kelvin
# submission_df['FFV'] **= 2
submission_df.to_csv("submission.csv", index=False)
print("Submission saved as 'submission.csv'.")

print("\n--- OOF MAE Summary (Ensembled with Hypertuning) ---")
for target, oof in oof_predictions.items():
    if target == "Tg":
        train_df = train_tg2.copy()
    elif target == "Tc":
        train_df = train_tc.copy()
    elif target == "FFV":
        train_df = train_ffv.copy()
    elif target == "Density":
        train_df = train_density.copy()
    elif target == "Rg":
        train_df = train_rg.copy()
    y_true = train_df.dropna(subset=[target])[target]
    print(f"{target}: MAE = {mean_absolute_error(y_true, oof):.4f}")

print(f"\nTotal training time: {time.time() - start_time:.2f} seconds.")


submission_df




