import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')


import kagglehub
path = kagglehub.dataset_download("senkin13/rdkit-2025-3-3-cp311")
print("Path to dataset files", path)


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem


X="We ballin now"
X


df1=pd.read_csv('/kaggle/input/neurips-dataset/tg.csv')


y = df1.iloc[:, 0]
X = df1.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)  #69? why not?


model1 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,        
    max_depth=3,             
    subsample=0.8,                
    reg_alpha=0.1,           
    random_state=69
)
model1.fit(X_train,y_train)


PredictionVal=model1.predict(X_test)
predictionTra=model1.predict(X_train)


MeaVal1=mean_absolute_error(PredictionVal,y_test)
MeaTra1=mean_absolute_error(predictionTra,y_train)


print(f"Validation Error: {MeaVal1}")
print(f"Training Error: {MeaTra1}")


print("1/5 Targets completed!! ðŸŽŠ")


df2=pd.read_csv('/kaggle/input/neurips-dataset/ffv.csv')


y = df2.iloc[:, 0]
X = df2.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)


model2 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,        
    max_depth=3,             
    subsample=0.8,                
    reg_alpha=0.1,           
    random_state=69
)
model2.fit(X_train,y_train)


PredictionVal=model2.predict(X_test)
predictionTra=model2.predict(X_train)


MeaVal2=mean_absolute_error(PredictionVal,y_test)
MeaTra2=mean_absolute_error(predictionTra,y_train)


print(f"Validation Error: {MeaVal2}")
print(f"Training Error: {MeaTra2}")


df3=pd.read_csv('/kaggle/input/neurips-dataset/tc.csv')


y = df3.iloc[:, 0]
X = df3.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)


model3 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,        
    max_depth=3,             
    subsample=0.8,                
    reg_alpha=0.1,           
    random_state=69
)
model3.fit(X_train,y_train)


PredictionVal=model3.predict(X_test)
predictionTra=model3.predict(X_train)


MeaVal3=mean_absolute_error(PredictionVal,y_test)
MeaTra3=mean_absolute_error(predictionTra,y_train)


print(f"Validation Error: {MeaVal3}")
print(f"Training Error: {MeaTra3}")


df4=pd.read_csv('/kaggle/input/neurips-dataset/density.csv')


y = df4.iloc[:, 0]
X = df4.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)


model4 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,        
    max_depth=3,             
    subsample=0.8,                
    reg_alpha=0.1,           
    random_state=69
)
model4.fit(X_train,y_train)


PredictionVal=model4.predict(X_test)
predictionTra=model4.predict(X_train)


MeaVal4=mean_absolute_error(PredictionVal,y_test)
MeaTra4=mean_absolute_error(predictionTra,y_train)


print(f"Validation Error: {MeaVal4}")
print(f"Training Error: {MeaTra4}")


df5=pd.read_csv('/kaggle/input/neurips-dataset/rg.csv')


y = df5.iloc[:, 0]
X = df5.iloc[:, 1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)


model5 = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,        
    max_depth=3,             
    subsample=0.8,                
    reg_alpha=0.1,           
    random_state=69
)
model5.fit(X_train,y_train)


PredictionVal=model5.predict(X_test)
predictionTra=model5.predict(X_train)


MeaVal5=mean_absolute_error(PredictionVal,y_test)
MeaTra5=mean_absolute_error(predictionTra,y_train)


print(f"Validation Error: {MeaVal5}")
print(f"Training Error: {MeaTra5}")


Test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv', dtype=str)


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

Test = pd.concat([Test, preprocessing(Test)], axis=1)
Test['Ipc']=np.log10(Test['Ipc'])

Test=Test.drop(['id','SMILES'],axis=1)


pred1=model1.predict(Test)
pred2=model2.predict(Test)
pred3=model3.predict(Test)
pred4=model4.predict(Test)
pred5=model5.predict(Test)


ss=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')


ss['Tg']=pred1
ss['FFV']=pred2
ss['Tc']=pred3
ss['Density']=pred4
ss['Rg']=pred5


ss.to_csv('submission.csv',index =False)




