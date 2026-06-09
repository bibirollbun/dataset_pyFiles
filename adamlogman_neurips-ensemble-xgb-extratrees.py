!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl
!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor



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


# tg_=pd.read_csv('/kaggle/input/new-dataset/tg.csv')
# rg_=pd.read_csv('/kaggle/input/new-dataset/rg.csv')
# tc_=pd.read_csv('/kaggle/input/new-dataset/tc.csv')
# ffv_=pd.read_csv('/kaggle/input/new-dataset/ffv.csv')
# density_=pd.read_csv('/kaggle/input/new-dataset/density.csv')
# test_=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')


tg=pd.read_csv('/kaggle/input/modred-dataset/desc_tg.csv')
tc=pd.read_csv('/kaggle/input/modred-dataset/desc_tc.csv')
rg=pd.read_csv('/kaggle/input/modred-dataset/desc_rg.csv')
ffv=pd.read_csv('/kaggle/input/modred-dataset/desc_ffv.csv')
density=pd.read_csv('/kaggle/input/desc-de-dataset/desc_de.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ID=test['id']


# tg plt.xlim(-223, 480)
# tg_ plt.xlim(-223, 492)
# tc plt.xlim(0, 0.54)
# rg plt.xlim(0, 33)
# ffv plt.xlim(0.3034, 0.426)
# # plt.xlim(0, 1.795)
## plt.xlim(0, 1.617)


# plt.figure(figsize=(10,4))
# # plt.xlim(0, 1.564)
# sns.boxplot(x=de_filtered_.Density)



for i in (tg,tc,rg,ffv,density):
     i.drop(columns=[col for col in i.columns if i[col].nunique() == 1],axis=1,inplace=True)


# Remove columns with object or category dtype
tg = tg.select_dtypes(exclude=['object', 'category'])
rg = rg.select_dtypes(exclude=['object', 'category'])
ffv = ffv.select_dtypes(exclude=['object', 'category'])
tc = tc.select_dtypes(exclude=['object', 'category'])
density  = density.select_dtypes(exclude=['object', 'category'])


mols_test = [Chem.MolFromSmiles(s) for s in test_.SMILES]

# Initialize the Mordred Calculator
calc = Calculator(descriptors, ignore_3D=True) # ignore_3D=True for 2D descriptors

desc_test = calc.pandas(mols_test)


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


# # Assume your DataFrame is called df and the target column is named 'target'
# tg_filtered = tg[(tg['Tg'] > -223) & (tg['Tg'] < 480)]
# tc_filtered = tc[(tc['Tc'] > 0) & (tc['Tc'] < 0.54)]
# rg_filtered = rg[(rg['Rg'] > 0) & (rg['Rg'] < 33)]
# ffv_filtered = ffv[(ffv['FFV'] > 0.307) & (ffv['FFV'] <  0.42)]
# de_filtered = density[(density['Density'] > 0) & (density['Density'] < 1.76)]

# # Assume your DataFrame is called df and the target column is named 'target'
# tg_filtered_ = tg_[(tg_['Tg'] > -223) & (tg_['Tg'] < 492)]
# tc_filtered_ = tc_[(tc_['Tc'] > 0) & (tc_['Tc'] < 0.54)]
# rg_filtered_ = rg_[(rg_['Rg'] > 0) & (rg_['Rg'] < 33)]
# ffv_filtered_ = ffv_[(ffv_['FFV'] > 0.307) & (ffv_['FFV'] <  0.42)]
# de_filtered_ = density_[(density_['Density'] > 0) & (density_['Density'] < 1.564)]


Model_tg= XGBRegressor(n_estimators= 2173,random_state=42, learning_rate= 0.0672418745539774, max_depth= 6, reg_lambda= 5.545520219149715)
Model_rg = XGBRegressor(n_estimators= 520,random_state=42, learning_rate= 0.07324113948440986, max_depth= 5, reg_lambda=0.9717380315982088)
Model_ffv = XGBRegressor(n_estimators= 2202,random_state=42, learning_rate= 0.07220580588586338, max_depth= 4, reg_lambda= 2.8872976032666493)
Model_tc = XGBRegressor(n_estimators= 1488,random_state=42, learning_rate= 0.010456188013762864, max_depth= 5, reg_lambda= 9.970345982204618)
Model_de = XGBRegressor(n_estimators= 1958,random_state=42, learning_rate= 0.10955287548172478, max_depth= 5, reg_lambda= 3.074470087965767)



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


# Let’s define a reusable function to train and evaluate our machine learning model.

def model_seed_1(train_d,test_d,model,target,submission=False):
    # We divide the data into training and validation sets for model evaluation
    X=train_d.drop(target,axis=1)
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model(random_state=21)
    if submission==False:
       Model.fit(X_train,y_train)
       y_pred=Model.predict(X_test)
       return mean_absolute_error(y_pred,y_test)         # We assess our model performance using MAE metric
    if submission==True:
       Model.fit(X,y)
       submission=Model.predict(test_d)
       return submission
def model_seed_2(train_d,test_d,model,target,submission=False):
    # We divide the data into training and validation sets for model evaluation
    X=train_d.drop(target,axis=1)
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model(random_state=42)
    if submission==False:
       Model.fit(X_train,y_train)
       y_pred=Model.predict(X_test)
       return mean_absolute_error(y_pred,y_test)         # We assess our model performance using MAE metric
    if submission==True:
       Model.fit(X,y)
       submission=Model.predict(test_d)
       return submission 
def model_seed_3(train_d,test_d,model,target,submission=False):
    # We divide the data into training and validation sets for model evaluation
    X=train_d.drop(target,axis=1)
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model(random_state=666)
    if submission==False:
       Model.fit(X_train,y_train)
       y_pred=Model.predict(X_test)
       return mean_absolute_error(y_pred,y_test)         # We assess our model performance using MAE metric
    if submission==True:
       Model.fit(X,y)
       submission=Model.predict(test_d)
       return submission 





# # We divide the data into training and validation sets for model evaluation
# train_cols = set(tg_filtered.columns) - {'Tg'}
# test_cols = set(desc_test.columns)
# # Intersect the feature columns
# common_cols = list(train_cols & test_cols)
# X=tg_filtered[common_cols].copy()
# y=tg_filtered['Tg'].copy()
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

# Model=Model_tg
# Model.fit(X_train,y_train)
# y_pred=Model.predict(X_test)
# mean_absolute_error(y_pred,y_test) 
# # Actual vs Predicted plot
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Actual vs. Predicted Values")
# plt.show()



# X=df_filtered.drop('Tg',axis=1)
# y=df_filtered['Tg'].copy()
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

# Model_1=ExtraTreesRegressor(random_state=21)
# Model_1.fit(X_train,y_train)
# y_pred_1=Model_1.predict(X_test)
# mean_absolute_error(y_pred_1,y_test)         # We assess our model performance using MAE metric



# # Actual vs Predicted plot
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_1, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Actual vs. Predicted Values")
# plt.show()



# Model_2=ExtraTreesRegressor(random_state=42)
# Model_2.fit(X_train,y_train)
# y_pred_2=Model_2.predict(X_test)
# mean_absolute_error(y_pred_2,y_test)         # We assess our model performance using MAE metric



# # Actual vs Predicted plot
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_2, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Actual vs. Predicted Values")
# plt.show()



# Model_3=ExtraTreesRegressor(random_state=666)
# Model_3.fit(X_train,y_train)
# y_pred_3=Model_3.predict(X_test)
# mean_absolute_error(y_pred_3,y_test)         # We assess our model performance using MAE metric



# # Actual vs Predicted plot
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred_3, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Actual vs. Predicted Values")
# plt.show()



# y_pred=(y_pred_1+y_pred_2+y_pred_3)/3


# # Actual vs Predicted plot
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel("Actual")
# plt.ylabel("Predicted")
# plt.title("Actual vs. Predicted Values")
# plt.show()



# # Average predictions from 3 model seeds for each target
# tg_result = (
#     model_seed_1(tg_, test, ExtraTreesRegressor, 'Tg', submission=True) +
#     model_seed_2(tg_, test, ExtraTreesRegressor, 'Tg', submission=True) +
#     model_seed_3(tg_, test, ExtraTreesRegressor, 'Tg', submission=True)
# ) / 3

# ffv_result = (
#     model_seed_1(ffv_, test, ExtraTreesRegressor, 'FFV', submission=True) +
#     model_seed_2(ffv_, test, ExtraTreesRegressor, 'FFV', submission=True) +
#     model_seed_3(ffv_, test, ExtraTreesRegressor, 'FFV', submission=True)
# ) / 3

# tc_result = (
#     model_seed_1(tc_, test, ExtraTreesRegressor, 'Tc', submission=True) +
#     model_seed_2(tc_, test, ExtraTreesRegressor, 'Tc', submission=True) +
#     model_seed_3(tc_, test, ExtraTreesRegressor, 'Tc', submission=True)
# ) / 3

# density_result = (
#     model_seed_1(density_, test, ExtraTreesRegressor, 'Density', submission=True) +
#     model_seed_2(density_, test, ExtraTreesRegressor, 'Density', submission=True) +
#     model_seed_3(density_, test, ExtraTreesRegressor, 'Density', submission=True)
# ) / 3

# rg_result = (
#     model_seed_1(rg_, test, ExtraTreesRegressor, 'Rg', submission=True) +
#     model_seed_2(rg_, test, ExtraTreesRegressor, 'Rg', submission=True) +
#     model_seed_3(rg_, test, ExtraTreesRegressor, 'Rg', submission=True)
# ) / 3


#  # Finally, we use the model to predict on the test set and prepare the submission file.

# sub={'id':ID,'Tg':tg_result,
#      'FFV':ffv_result,
#      'Tc':tc_result,
#      'Density':density_result,
#      'Rg':rg_result}


 # Finally, we use the model to predict on the test set and prepare the submission file.

sub_={'id':ID,'Tg':model(tg,desc_test,Model_tg,'Tg',submission=True),
     'FFV':model(ffv,desc_test,Model_ffv,'FFV',submission=True),
     'Tc':model(tc,desc_test,Model_tc,'Tc',submission=True),
     'Density':model(density,desc_test,Model_de,'Density',submission=True),
     'Rg':model(rg,desc_test,Model_rg,'Rg',submission=True)}
 # Finally, we use the model to predict on the test set and prepare the submission file.

# sub_={'id':ID,'Tg':model(tg_,test,ExtraTreesRegressor(random_state=42,),'Tg',submission=True),
#      'FFV':model(ffv_,test,ExtraTreesRegressor(random_state=42,),'FFV',submission=True),
#      'Tc':model(tc_,test,ExtraTreesRegressor(random_state=42,),'Tc',submission=True),
#      'Density':model(density_,test,ExtraTreesRegressor(random_state=42,),'Density',submission=True),
#      'Rg':model(rg_,test,ExtraTreesRegressor(random_state=42,),'Rg',submission=True)}


# submission=pd.DataFrame(sub)
submission_=pd.DataFrame(sub_)


# Select numeric columns (excluding 'id')
features = [col for col in submission.columns if col != 'id']

# Compute average
df_avg = submission.copy()
df_avg[features] = (submission[features] + submission_[features]) / 2


df_avg


submission_.to_csv('submission.csv',index=False)

