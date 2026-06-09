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


# install RDKit for offline use
!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from typing import Union,Callable

from rdkit import Chem


TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
SMILES = 'SMILES'


def clean_smiles(smiles:str)-> Union[str,None]:
    # smiles who have these patterns will cause Chem parse ERROR
    bad_patterns = [
            '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
            "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
            # Additional patterns that cause issues
            '([R])', '([R1])', '([R2])', 
        ]
    for bad in bad_patterns:
        if bad in smiles:
            return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        usmile = Chem.MolToSmiles(mol,canonical=True)
        return usmile
    except Exception as e:
        print(e)
        return None


def preprocessing_datas(
        df:pd.DataFrame,
        target:str='all',
        targetName:Union[str,None]=None,
        preFuncs:Union[Callable,None]=None)-> pd.DataFrame:
    """
    target:the target which this df have
    targetName:the real name of the target in the df
    preFuncs:function to modify the target data
    """
    df[SMILES] = df[SMILES].apply(clean_smiles)
    # 去掉SMILES是nan的
    df = df[df[SMILES].notna()].reset_index(drop=True)
    if target == 'all':
        # modify train data
        df = df.drop('id',axis=1)
        return df
    elif target in TARGETS:
        # modify extra data
        if targetName is None:
            raise ValueError("targetName must be given.")
        df = df.rename(columns={targetName:target})
        # 去掉target是nan的
        df = df[df[target].notna()].reset_index(drop=True)
        # 提取SMILE和target列
        # df = df[[SMILES,target]]
        # 将as_index设置为True的话，groupby的列会变成index
        df = df.groupby(SMILES,as_index=False)[target].mean()
        if preFuncs is not None:
            df[target] = df[target].apply(preFuncs)
        return df
    else:
        raise ValueError(f"target should in {TARGETS+['all']}")


def merge_extra_data(train:pd.DataFrame,
                     extra:pd.DataFrame,
                     target:str,
                     rsuffix:str='_right')->pd.DataFrame:
    # 连接train和extra数据
    df = pd.merge(train,extra,how='outer',on=SMILES,suffixes=('',rsuffix))
    nanIndex = df[df[target].isna()].index
    df.loc[nanIndex,target] = df.loc[nanIndex,f'{target}{rsuffix}']
    df = df.drop(f'{target}{rsuffix}',axis=1)
    return df


dfTrain = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
dfTrain = preprocessing_datas(dfTrain)


dfTrain.info()


df = pd.read_excel('/kaggle/input/smiles-extra-data/data_dnst1.xlsx')
needDrop = df[df['density(g/cm3)']=='nylon'].index
df = df.drop(needDrop,axis=0)


df[df['density(g/cm3)']=='nylon']


df.to_excel("/kaggle/data_dnst1.xlsx")


# extra数据的路径，以及对应的preprocessing参数
filePaths = [
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv',
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv',
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv',
    '/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv',
    '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    '/kaggle/data_dnst1.xlsx'
]
fileTargets = [
    ('Tg','Tg (C)'),
    ('Tc','TC_mean'),
    ('Tg','Tg'),
    ('FFV','FFV'),
    ('Tc','TC_mean'),
    ('Tg','Tg'),
    ('Tg','Tg [K]'),
    ('Density','density(g/cm3)')
]
fileFuncs = [
    None,
    None,
    None,
    None,
    None,
    None,
    lambda x: x - 273.15,
    None
]

# 开始合并
for file,(target,targetName),preFunc in zip(filePaths,fileTargets,fileFuncs):
    print(f'read file:{file}')
    if file.endswith('.csv'):
        dfExtra = pd.read_csv(file)
    elif file.endswith('.xlsx'):
        dfExtra = pd.read_excel(file)
    
    print('preprocessing data')
    dfExtra = preprocessing_datas(dfExtra,
                                  target=target,
                                  targetName=targetName,
                                  preFuncs=preFunc)
    print('merging')
    dfTrain = merge_extra_data(
        dfTrain,
        dfExtra,
        target
    )


dfTrain.to_csv('/kaggle/mergedTrain.csv',index=False)


from tqdm import tqdm

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors,rdmolops
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator,\
    GetAtomPairGenerator,GetTopologicalTorsionGenerator

import networkx as nx


def augment_smiles_dataset(data:pd.DataFrame,numAugments:int = 3) -> pd.DataFrame:
    augSmiles = []
    total = data.shape[0]
    for i,row in tqdm(data.iterrows(),total=total):
        rowDict = dict(row)
        augSmiles.append(rowDict)
        mol = Chem.MolFromSmiles(rowDict['SMILES'])
        for _ in range(numAugments):
            randSmiles = Chem.MolToSmiles(mol,doRandom=True)
            augDict = rowDict.copy()
            augDict['SMILES'] = randSmiles
            augSmiles.append(augDict)
    
    augSmiles = pd.DataFrame(augSmiles)
    # duplicate
    augSmiles = augSmiles.groupby("SMILES",as_index=False).mean()

    return augSmiles


dfAug = augment_smiles_dataset(dfTrain)


dfAug.to_csv('/kaggle/augmentedTrain.csv')


def smiles_to_combined_fingerprints_with_descriptors(smiles,radius=2,n_bits=128):
    # 提取smiles特征（即指纹与描述）
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros(n_bits * 3 + 167),None,smiles,False
    # finger prints
    # morgan fingerprint
    generator = GetMorganGenerator(radius=radius,fpSize=n_bits)
    morgan_fp = generator.GetFingerprint(mol)
    # atom pair fingerprint
    atomPairGen = GetAtomPairGenerator(fpSize=n_bits)
    atomPair_fp = atomPairGen.GetFingerprint(mol)
    # torsion fingerprint
    torsionGen = GetTopologicalTorsionGenerator(fpSize=n_bits)
    torsion_fp = torsionGen.GetFingerprint(mol)
    # MACCSkeys
    maccs_fp = MACCSkeys.GenMACCSKeys(mol) # len 167
    # combine fingerprints
    combined_fp = np.concatenate([
        np.array(morgan_fp),
        np.array(atomPair_fp),
        np.array(torsion_fp),
        np.array(maccs_fp),
    ])

    # descriptors
    descriptor_values = {}
    for name,func in Descriptors.descList:
        try:
            descriptor_values[name] = func(mol)
        except:
            descriptor_values[name] = None

    descriptor_values['NumAtoms'] = mol.GetNumAtoms()
    descriptor_values['smiles'] = smiles

    # graph features
    adj = rdmolops.GetAdjacencyMatrix(mol) # 获取邻接矩阵(图的知识)
    G = nx.from_numpy_array(adj)
    # 添加三个图的参数：graph_diameter 图直径
    #                  avg_shortest_path 平均最短路径
    #                  num_cycles 循环数量
    if nx.is_connected(G):
        descriptor_values['graph_diameter'] = nx.diameter(G)
        descriptor_values['avg_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        descriptor_values['graph_diameter'] = 0
        descriptor_values['avg_shortest_path'] = 0

    descriptor_values['num_cycles'] = len(list(nx.cycle_basis(G)))

    return combined_fp,descriptor_values,smiles,True


def creat_features(smilesList,radius=2,n_bits=128):
    fingerprints = []
    descriptors = []
    invalidSMILES = []
    for smiles in tqdm(smilesList):
        fp,desc,valid_smiles,is_valid = \
            smiles_to_combined_fingerprints_with_descriptors(smiles,radius=radius,n_bits=n_bits)
        fingerprints.append(fp)
        descriptors.append(desc)
        if not is_valid:
            invalidSMILES.append(smiles)
    
    return np.array(fingerprints),descriptors,invalidSMILES

def merge_features_to_dataframe(smileList):
    fingerPrints,descriptors,invalidSMILES = \
        creat_features(smileList)
    
    _, fpNums = fingerPrints.shape
    columns = [f'fp{i:03d}' for i in range(fpNums)]
    dfFP = pd.DataFrame(fingerPrints,columns=columns)

    dfDesc = pd.DataFrame(descriptors)
    # 需要去掉的列，因为一列都是nan
    needDropCols = ['MaxPartialCharge',
                    'MinPartialCharge',
                    'MaxAbsPartialCharge',
                    'MinAbsPartialCharge',
                    'BCUT2D_MWHI',
                    'BCUT2D_MWLOW',
                    'BCUT2D_CHGHI',
                    'BCUT2D_CHGLO',
                    'BCUT2D_LOGPHI',
                    'BCUT2D_LOGPLOW',
                    'BCUT2D_MRHI',
                    'BCUT2D_MRLOW']
    dfDesc = dfDesc.drop(needDropCols,axis=1)
    dfFeatured = pd.concat(
        [dfFP,dfDesc],
        axis=1)
    
    return dfFeatured


dfFeatured = merge_features_to_dataframe(dfAug['SMILES'])
dfFeatured = pd.concat([dfFeatured,dfAug[TARGETS]],axis=1)


dfFeatured.to_csv('/kaggle/featuredTrain.csv',index=False)


dfTest = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
testFeature = merge_features_to_dataframe(dfTest["SMILES"])
testFeature.to_csv('/kaggle/featuredTest.csv',index=False)


import pandas as pd
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold,\
    SelectFromModel
from sklearn.ensemble import RandomForestRegressor
def get_label_dataset(X:pd.DataFrame,y:pd.DataFrame,label:str)->Tuple[pd.DataFrame,pd.Series]:
    # 得到每个label非nan的数据行
    nonNanIndex = y[label].notna()
    X = X.loc[nonNanIndex,:]
    y = y.loc[nonNanIndex,label]

    return X,y

_canDropCorr = ['SMR_VSA7', 'ExactMolWt', 'fr_nitro_arom', 'NumHDonors',
 'fr_amide', 'NOCount', 'NumAromaticCarbocycles', 'fr_nitrile', 'NumSaturatedCarbocycles',
 'Chi0v', 'Chi2n', 'SlogP_VSA6', 'fr_C_O_noCOO', 'fr_halogen', 'avg_shortest_path',
 'RingCount', 'FpDensityMorgan3', 'graph_diameter', 'VSA_EState6', 'Chi1n', 'NumAmideBonds',
 'EState_VSA10', 'fr_alkyl_halide', 'fr_phenol_noOrthoHbond', 'MolMR', 'num_cycles',
 'NumAtoms', 'Chi3v', 'NumAromaticRings', 'SlogP_VSA5', 'fr_diazo', 'NumHAcceptors', 'NumUnspecifiedAtomStereoCenters',
 'Chi4v', 'fr_imide', 'fr_NH2', 'MolLogP', 'Phi', 'fr_Nhpyrrole', 'NumHeteroatoms', 'fr_COO2',
 'FpDensityMorgan2', 'Chi0n', 'MaxEStateIndex', 'NumValenceElectrons', 'Kappa1', 'VSA_EState10',
 'fr_Al_OH_noTert', 'fr_nitro_arom_nonortho', 'BertzCT', 'NumSaturatedRings', 'NumRotatableBonds',
 'Chi0', 'Chi4n', 'fr_phos_ester', 'HeavyAtomMolWt', 'Chi1', 'Kappa2', 'Chi2v', 'Chi1v',
 'Chi3n', 'HallKierAlpha', 'HeavyAtomCount', 'fr_phenol', 'LabuteASA', 'fr_C_O',
 'fr_unbrch_alkane', 'fr_benzene', 'PEOE_VSA14', 'fr_NH1', 'VSA_EState1']

class FeatureSelect:
    def __init__(self):
        self.sfmMap = {
            label:SelectFromModel(estimator=RandomForestRegressor()) 
            for label in TARGETS
            }
        self.scaler = MinMaxScaler()
        self.var_thresh = VarianceThreshold(threshold=0.01)
        self.var_thresh.set_output(transform='pandas')
    
    def fit(self,X:pd.DataFrame,y):
        '''
        fit 全部label
        '''
        # 通过皮尔逊相关系数去除列
        X = X.drop(_canDropCorr,axis=1)
        # fit scaler for col Ipc
        X['Ipc'] = self.scaler.fit_transform(X[['Ipc']])
        # 舍弃var很小的列
        X = self.var_thresh.fit_transform(X)
        # fit 随机森林筛选器
        for label in TARGETS:
            X_label,y_label = get_label_dataset(X,y,label)
            self.sfmMap[label].set_output(transform='pandas')
            self.sfmMap[label].fit(X_label,y_label)

    def fit_transform(self,X:pd.DataFrame,y,label):
        '''
        fit并transform单一label的数据
        不行，这个会重复训练
        '''
        if label not in TARGETS:
            raise ValueError("label must in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']")
        # 通过皮尔逊相关系数去除列
        X = X.drop(_canDropCorr,axis=1)
        # fit scaler for col Ipc
        X['Ipc'] = self.scaler.fit_transform(X[['Ipc']])
        # 舍弃var很小的列
        X = self.var_thresh.fit_transform(X)
        # fit 随机森林筛选器
        self.sfmMap[label].set_output(transform='pandas')
        X = self.sfmMap[label].fit_transform(X,y)

        return X

    def transform(self,X:pd.DataFrame,label):
        '''
        用已经fit好的模型trasform
        '''
        if label not in TARGETS:
            raise ValueError("label must in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']")
        # 通过皮尔逊相关系数去除列
        X = X.drop(_canDropCorr,axis=1)
        # fit scaler for col Ipc
        X['Ipc'] = self.scaler.transform(X[['Ipc']])
        # 舍弃var很小的列
        X = self.var_thresh.transform(X)
        # fit 随机森林筛选器
        X = self.sfmMap[label].transform(X)

        return X


from typing import Tuple

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split,cross_val_score,\
    KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from skopt import gp_minimize, space
from skopt.utils import use_named_args
from xgboost import XGBRegressor

import matplotlib.pyplot as plt

from tqdm import tqdm


# metric:https://www.kaggle.com/code/metric/open-polymer-2025/notebook
class ParticipantVisibleError(Exception):
    pass


# These values are from the train data.
MINMAX_DICT =  {
        'Tg': [-148.0297376, 472.25],
        'FFV': [0.2269924, 0.77709707],
        'Tc': [0.0465, 0.524],
        'Density': [0.748691234, 1.840998909],
        'Rg': [9.7283551, 34.672905605],
    }


def scaling_error(labels, preds, property):
    error = np.abs(labels - preds)
    min_val, max_val = MINMAX_DICT[property]
    label_range = max_val - min_val
    return np.mean(error / label_range)


def get_property_weights(labels):
    property_weight = []
    for property in MINMAX_DICT.keys():
        valid_num = np.sum(labels[property].notna())
        property_weight.append(valid_num)
    property_weight = np.array(property_weight)
    property_weight = np.sqrt(1 / property_weight)
    return (property_weight / np.sum(property_weight)) * len(property_weight)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute weighted Mean Absolute Error (wMAE) for the Open Polymer challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'id': unique identifier for each sequence
      - Columns 'Tg', 'FFV', 'Tc', 'Density', 'Rg' as the predicted targets

    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4})
    >>> submission = pd.DataFrame({'id': range(4), 'Tg': [0.5]*4, 'FFV': [0.5]*4, 'Tc': [0.5]*4, 'Density': [0.5]*4, 'Rg': [0.5]*4})
    >>> round(score(solution, submission, row_id_column_name=row_id_column_name), 4)
    0.2922
    >>> submission = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4} )
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    """
    chemical_properties = list(MINMAX_DICT.keys())
    property_maes = []
    property_weights = get_property_weights(solution[chemical_properties])
    for property in chemical_properties:
        is_labeled = solution[property].notna()
        property_maes.append(scaling_error(solution.loc[is_labeled, property], submission.loc[is_labeled, property], property))

    if len(property_maes) == 0:
        raise RuntimeError('No labels')
    return float(np.average(property_maes, weights=property_weights))


class XGBRegressorForFiveTargets:
    '''
    把预测5个target的model集合起来，其中的每一个model
    都对对应的target进行了调参。
    '''
    def __init__(self):
        self.featureSelector = FeatureSelect()
        self.tunedParam = {
            'Tg': {'learning_rate': 0.3435427812850668,
                'max_depth': 8,
                'reg_lambda': 10.0,
                'n_estimators': 333},
            'FFV': {'learning_rate': 0.3216722248147638,
                'max_depth': 8,
                'reg_lambda': 0.1,
                'n_estimators': 1999},
            'Tc': {'learning_rate': 0.13997398052504034,
                'max_depth': 8,
                'reg_lambda': 0.1,
                'n_estimators': 220},
            'Density': {'learning_rate': 0.4968579586896936,
                'max_depth': 8,
                'reg_lambda': 9.527175626562906,
                'n_estimators': 300},
            'Rg': {'learning_rate': 0.5,
                'max_depth': 8,
                'reg_lambda': 10.0,
                'n_estimators': 217}}
        self.models = {}
        for label in TARGETS:
            self.models[label] = XGBRegressor(**self.tunedParam[label])
    
    def fit(self,X:pd.DataFrame,y:pd.DataFrame):
        self.featureSelector.fit(X,y)
        for label in TARGETS:
            print(f"{label} training...")
            X_label,y_label = get_label_dataset(X,y,label)
            X_label = self.featureSelector.transform(X_label,label)
            self.models[label].fit(X_label,y_label)
    
    def predict(self,X:pd.DataFrame):
        pred = {}
        for label in TARGETS:
            X_label = self.featureSelector.transform(X,label)
            y_pred = self.models[label].predict(X_label)
            pred[label] = y_pred
        
        return pd.DataFrame(pred)
    
    def wmae_score(self,yPred:pd.DataFrame,yValid:pd.DataFrame):
        yPred = yPred.set_index(yValid.index)
        return score(yValid,yPred,'id')


dataset = pd.read_csv('/kaggle/featuredTrain.csv')
dataset = dataset.drop(['smiles'],axis=1) # Ipc数据数量级太大，不堪用，去掉


# X_train,X_test,y_train,y_test = train_test_split(
#     dataset.drop(TARGETS,axis=1),
#     dataset[TARGETS],
#     test_size=0.2,
#     random_state=42,
#     shuffle=True
#     )


# model = XGBRegressorForFiveTargets()
# model.fit(X_train,y_train)


# y_pred = model.predict(X_test)
# model.wmae_score(y_pred,y_test)


# 把X和y分开
X = dataset.drop(TARGETS, axis=1)
y = dataset[TARGETS]
model = XGBRegressorForFiveTargets()
model.fit(X,y)


# pred = model.predict(X)
# print(model.wmae_score(pred,y))


test = pd.read_csv('/kaggle/featuredTest.csv')
test = test.drop(['smiles'],axis=1) # Ipc数据数量级太大，不堪用，去掉
pred = model.predict(test)


Otest = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
submission = pd.concat([Otest,pred],axis=1)
submission = submission.drop('SMILES',axis=1)
submission.to_csv("submission.csv",index=False)

