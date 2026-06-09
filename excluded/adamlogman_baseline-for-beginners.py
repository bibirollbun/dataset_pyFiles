!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


 # Importing Required Libraries\nLet's begin by importing the essential Python libraries needed for data processing, visualization, and modeling.

import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


 #We will load both the training and test datasets using pandas, and store test IDs 

train=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')
test=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
ss=pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
ID=test['id'].copy()


train.head()


# Get all available RDKit descriptors
descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]

# Function to compute all descriptors
def compute_all_descriptors(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    descriptor_values = {}
    for name, func in Descriptors.descList:
        try:
            descriptor_values[name] = func(mol)
        except:
            descriptor_values[name] = None  # In case a descriptor fails

    descriptor_values['SMILES'] = smile
    return descriptor_values

# Apply to all SMILES
data_t = [compute_all_descriptors(smi) for smi in train['SMILES']]
data_ts=[compute_all_descriptors(smi) for smi in test['SMILES']]
data_t = [d for d in data_t if d is not None]
data_ts = [d for d in data_ts if d is not None]
train_df = pd.DataFrame(data_t)
test_df = pd.DataFrame(data_ts)

# Move SMILES column to the front
cols_t = ['SMILES'] + [c for c in train_df.columns if c != 'SMILES']
cols_ts = ['SMILES'] + [c for c in test_df.columns if c != 'SMILES']
train_df = train_df[cols_t]
test_df = test_df[cols_ts]




# Here we will merge old train with trian_df which have new descriptor features 
train=train.merge(train_df,on='SMILES',how='left')
test=test.merge(test_df,on='SMILES',how='left')


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


# we'll drop certain descriptors (features) that contain missing values across the dataset
test=test.drop( ['id','BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','SMILES'],axis=1)
train=train.drop( ['id','BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','Tg','FFV','Tc','Density','Rg'],axis=1)


tg=t_1.merge(train,on='SMILES',how='left')
ffv=t_2.merge(train,on='SMILES',how='left')
tc=t_3.merge(train,on='SMILES',how='left')
density=t_4.merge(train,on='SMILES',how='left')
rg=t_5.merge(train,on='SMILES',how='left')


tg.shape,ffv.shape ,tc.shape ,density.shape ,rg.shape


tg = tg.replace([-np.inf, np.inf], np.nan)
ffv = ffv.replace([-np.inf, np.inf], np.nan)
tc = tc.replace([-np.inf, np.inf], np.nan)
rg = rg.replace([-np.inf, np.inf], np.nan)
density = density.replace([-np.inf, np.inf], np.nan)


tg.shape,ffv.shape ,tc.shape ,density.shape ,rg.shape


for i in (tg,tc,density,ffv,rg):
    i.drop('SMILES',axis=1,inplace=True)
    i.dropna(inplace=True)


# Letâ€™s define a reusable function to train and evaluate our machine learning model.
from sklearn.ensemble import ExtraTreesRegressor

def model(train_d,test_d,model,target,submission=False):
    # We divide the data into training and validation sets for model evaluation
    X=train_d.drop([target],axis=1)
    y=train_d[target].copy()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

    Model=model( )
    if submission==False:
       Model.fit(X_train,y_train)
       y_pred=Model.predict(X_test)
       return mean_absolute_error(y_pred,y_test)         # We assess our model performance using MAE metric
    if submission==True:
       Model.fit(X,y)
       submission=Model.predict(test_d)
       return submission
        


model(tg,test,CatBoostRegressor,'Tg',submission=False)


model(ffv,test,CatBoostRegressor,'FFV',submission=False)


model(tc,test,ExtraTreesRegressor,'Tc',submission=False)


model(density,test,ExtraTreesRegressor,'Density',submission=False)


model(rg,test,ExtraTreesRegressor,'Rg',submission=False)


 # Finally, we use the model to predict on the test set and prepare the submission file.

sub={'id':ID,'Tg':model(tg,test,CatBoostRegressor,'Tg',submission=True),
     'FFV':model(ffv,test,CatBoostRegressor,'FFV',submission=True),
     'Tc':model(tc,test,ExtraTreesRegressor,'Tc',submission=True),
     'Density':model(density,test,ExtraTreesRegressor,'Density',submission=True),
     'Rg':model(rg,test,ExtraTreesRegressor,'Rg',submission=True)}


submission=pd.DataFrame(sub)


submission


submission.to_csv('submission.csv',index=False)

