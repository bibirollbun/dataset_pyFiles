!pip install /kaggle/input/rdkit-2025-3-3/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


import pandas as pd
import numpy as np

from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from catboost import CatBoostRegressor


data_path = "/kaggle/input/neurips-open-polymer-prediction-2025"

train = pd.read_csv(data_path + "/train.csv")
test = pd.read_csv(data_path + "/test.csv")
submission = pd.read_csv(data_path + "/sample_submission.csv")
ID = test['id'].copy()


train.head()


descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]

def compute_all_descriptors(smile) :
    mol = Chem.MolFromSmiles(smile)
    if mol is None :
        return None

    descriptor_values = {}
    for name, func in Descriptors.descList :
        try : 
            descriptor_values[name] = func(mol)
        except :
            descriptor_values[name] = None

    descriptor_values['SMILES'] = smile
    return descriptor_values


train_data = [compute_all_descriptors(smile) for smile in train['SMILES']]
train_data = [data for data in train_data if data is not None]

test_data = [compute_all_descriptors(smile) for smile in test['SMILES']]
test_data = [data for data in test_data if data is not None]


train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


train_df.head()


train=train.merge(train_df,on='SMILES',how='left')
test=test.merge(test_df,on='SMILES',how='left')


train.columns = [col.replace('_x', '').replace('_y', '') for col in train.columns]


train = train.loc[:, ~train.columns.duplicated()]


test.columns = [col.replace('_x', '').replace('_y', '') for col in test.columns]
test = test.loc[:, ~test.columns.duplicated()]


target_1 = train[['SMILES', 'Tg']].copy()
target_2 = train[['SMILES', 'FFV']].copy()
target_3 = train[['SMILES', 'Tc']].copy()
target_4 = train[['SMILES', 'Density']].copy()
target_5 = train[['SMILES', 'Rg']].copy()

target_1.dropna(inplace=True)
target_2.dropna(inplace=True)
target_3.dropna(inplace=True)
target_4.dropna(inplace=True)
target_5.dropna(inplace=True)


test=test.drop( ['id','BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','SMILES'],axis=1)
train=train.drop( ['id','BCUT2D_MWLOW','BCUT2D_MWHI','BCUT2D_CHGHI','BCUT2D_CHGLO','BCUT2D_LOGPHI','BCUT2D_LOGPLOW','BCUT2D_MRLOW','BCUT2D_MRHI','MinAbsPartialCharge','MaxPartialCharge','MinPartialCharge','MaxAbsPartialCharge','Tg','FFV','Tc','Density','Rg'],axis=1)

tg = target_1.merge(train, on='SMILES', how='left')
ffv = target_2.merge(train, on='SMILES', how='left')
tc = target_3.merge(train, on='SMILES', how='left')
density = target_4.merge(train, on='SMILES', how='left')
rg = target_5.merge(train, on='SMILES', how='left')


tg = tg.replace([-np.inf, np.inf], np.nan)
ffv = ffv.replace([-np.inf, np.inf], np.nan)
tc = tc.replace([-np.inf, np.inf], np.nan)
density = density.replace([-np.inf, np.inf], np.nan)
rg = rg.replace([-np.inf, np.inf], np.nan)


for i in (tg, tc, density, ffv, rg) :
    i.drop('SMILES', axis=1, inplace=True)
    i.dropna(inplace=True)


def model(train, test, model, target, submission=False) :
    X = train.drop([target], axis=1)
    y = train[target].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    Model = model()
    if submission == False :
        Model.fit(X_train, y_train)
        y_pred = Model.predict(X_test)
        return mean_squared_error(y_pred,y_test)
    if submission == True :
        Model.fit(X, y)
        submission=Model.predict(test)
        return submission


model(tg, test, CatBoostRegressor, 'Tg', submission=False)


model(ffv, test, CatBoostRegressor, 'FFV', submission=False)


from sklearn.ensemble import ExtraTreesRegressor

model(tc, test, ExtraTreesRegressor, 'Tc', submission=False)
model(density, test, ExtraTreesRegressor, 'Density')
model(rg, test, ExtraTreesRegressor, 'Rg', submission=False)


sub = {'id' : ID, 'Tg' : model(tg, test, CatBoostRegressor, 'Tg', submission=True),
       'FFV' : model(ffv, test, CatBoostRegressor, 'FFV', submission=True),
       'Tc' : model(tc, test, ExtraTreesRegressor, 'Tc', submission=True),
       'Density' : model(density, test, ExtraTreesRegressor, 'Density', submission=True),
       'Rg' : model(rg, test, ExtraTreesRegressor, 'Rg', submission=True),
}


submission=pd.DataFrame(sub)


submission


submission.to_csv('submission.csv', index=False)

