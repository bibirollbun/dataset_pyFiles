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


import rdkit
print(rdkit.__version__)


# Import Data

train_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv')

test_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')

dataset1_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv') # Tc
dataset1_df = dataset1_df.rename(columns={'TC_mean': 'Tc'})

dataset3_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv') # Tg

dataset4_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv') # FFV



train_df = pd.concat([train_df,
                     dataset1_df,
                     dataset3_df,
                     dataset4_df], axis=0).reset_index(drop=True)


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors








def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    numberOfAtoms = mol.GetNumAtoms()
    DP = round(600/numberOfAtoms)

    morgan = AllChem.GetMorganGenerator(radius=2, fpSize=1024)
    object = morgan.GetCountFingerprint(mol)
   
    fp = np.zeros((0,), dtype=np.int16)
    DataStructs.ConvertToNumpyArray(object, fp)
    return DP*fp




#X_train_feats = train_df["SMILES"].apply(smiles_to_fp)
X_train_feats = pd.DataFrame(np.vstack([smiles_to_fp(s) for s in train_df["SMILES"]]))
X_test_feats  = pd.DataFrame(np.vstack([smiles_to_fp(s) for s in test_df["SMILES"]]))







y_train = train_df[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].to_numpy()



from xgboost import XGBRegressor


sub_df = pd.read_csv('/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv')
y_pred = np.zeros_like(sub_df).astype(float)
task_names = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

XGB_PARAMS = {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 6, 'subsample': 0.7, 'colsample_bytree': 0.6, 'random_state': 42, 'n_jobs': -1, 'tree_method': 'hist'}





# Predict FFV

from sklearn.decomposition import PCA

y_col = y_train[:, 1]
mask  = ~np.isnan(y_col)

FFV_pca = PCA(n_components=500)
FFV_X_train_feats = FFV_pca.fit(X_train_feats[mask]).transform(X_train_feats)

# Create XGBoost classifier
FFV_model = XGBRegressor(**XGB_PARAMS)
FFV_model.fit(FFV_X_train_feats[mask], y_col[mask])

# Predict on test set
FFV_X_test_feats = FFV_pca.transform(X_test_feats)
y_pred[:, 1] = FFV_model.predict(FFV_X_test_feats)




# Predict Density

y_col = y_train[:, 3]
mask  = ~np.isnan(y_col)

Density_pca = PCA(n_components=250)
Density_X_train_feats = Density_pca.fit(X_train_feats[mask]).transform(X_train_feats)

Density_model = XGBRegressor(**XGB_PARAMS)
Density_model.fit(Density_X_train_feats[mask], y_col[mask])
# Predict on test set
Density_X_test_feats = Density_pca.transform(X_test_feats)
y_pred[:, 3] = Density_model.predict(Density_X_test_feats)






# Predict Rg

y_col = y_train[:, 4]
mask  = ~np.isnan(y_col)

Rg_pca = PCA(n_components=250)
Rg_X_train_feats = Rg_pca.fit(X_train_feats[mask]).transform(X_train_feats)

Rg_model = XGBRegressor(**XGB_PARAMS)
Rg_model.fit(Rg_X_train_feats[mask], y_col[mask])
# Predict on test set
Rg_X_test_feats = Rg_pca.transform(X_test_feats)
y_pred[:, 4] = Rg_model.predict(Rg_X_test_feats)




# Predict Tc

y_col = y_train[:, 2]
mask  = ~np.isnan(y_col)

Tc_pca = PCA(n_components=250)
Tc_X_train_feats = Tc_pca.fit(X_train_feats[mask]).transform(X_train_feats)

Tc_model = XGBRegressor(**XGB_PARAMS)
Tc_model.fit(Tc_X_train_feats[mask], y_col[mask])
# Predict on test set
Tc_X_test_feats = Tc_pca.transform(X_test_feats)
y_pred[:, 2] = Tc_model.predict(Tc_X_test_feats)


# Predict Tg

y_col = y_train[:, 0]
mask  = ~np.isnan(y_col)

Tg_pca = PCA(n_components=250)
Tg_X_train_feats = Tg_pca.fit(X_train_feats[mask]).transform(X_train_feats)

Tg_model = XGBRegressor(**XGB_PARAMS)
Tg_model.fit(Tg_X_train_feats[mask], y_col[mask])
# Predict on test set
Tg_X_test_feats = Tg_pca.transform(X_test_feats)
y_pred[:, 0] = Tg_model.predict(Tg_X_test_feats)



y_pred_df = pd.DataFrame(y_pred, columns =['Tg', 'FFV', 'Tc', 'Density', 'Rg', 'MW'])
y_pred_df
#sub_df


sub_df['Tg'] = y_pred_df['Tg']
sub_df['FFV'] =y_pred_df['FFV']
sub_df['Tc'] = y_pred_df['Tc']
sub_df['Density'] =y_pred_df['Density'] 
sub_df['Rg'] = y_pred_df['Rg']





#def MW(smiles):
 #   mol = Chem.MolFromSmiles(smiles)
  #  numberOfAtoms = mol.GetNumAtoms()
   # DP = round(600/numberOfAtoms)
    #return DP*Descriptors.MolWt(mol)


# for iteration
#Train_iter1 = train_df.copy().drop(["id", "SMILES"], axis=1)
#Train_iter1['MW']=pd.Series(train_df["SMILES"].apply(MW))

#y_pred_df['MW']=pd.Series(test_df["SMILES"].apply(MW))


#Train_iter1.head()





# Predict Density 2

#y_col = y_train[:, 3]
#mask  = ~np.isnan(y_col)

#TrainDensity1 = Train_iter1.dropna(subset = ['Density'])
#Train_FFV_pred = pd.Series(FFV_model.predict(X_train_feats[mask]))
#TrainDensity1['FFV'].fillna(Train_FFV_pred, inplace=True)

#Train_Density_pred = pd.Series(Density_model.predict(X_train_feats[mask]))
#TrainDensity1['Density'].fillna(Train_Density_pred, inplace=True)

#Train_Rg_pred = pd.Series(Rg_model.predict(X_train_feats[mask]))
#TrainDensity1['Rg'].fillna(Train_Rg_pred, inplace=True)

#Train_Tc_pred = pd.Series(Tc_model.predict(X_train_feats[mask]))
#TrainDensity1['Tc'].fillna(Train_Tc_pred, inplace=True)

#Train_Tg_pred = pd.Series(Tg_model.predict(X_train_feats[mask]))
#TrainDensity1['Tg'].fillna(Train_Tg_pred, inplace=True)





#X = TrainDensity1.copy()
#y = X.pop('Density')

#Density_model_2 = XGBRegressor(n_estimators=50, learning_rate=0.05)
#Density_model_2.fit(X, y)
# Predict on test set
#Test_Density = y_pred_df.drop('Density', axis=1)
#Test_Density['MW']=pd.Series(test_df["SMILES"].apply(MW))
#y_pred[:, 3] = Density_model_2.predict(Test_Density)
#sub_df['Density'] = pd.Series(y_pred[:, 3])



#sns.scatterplot(x=MW_train_Density, y=train_df['Density'])
#sns.scatterplot(x=y_pred[:, 1], y=y_pred[:, 0])



sub_df.to_csv('submission.csv', index=False)
print(sub_df)

