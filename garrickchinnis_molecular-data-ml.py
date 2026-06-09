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


#XGBoost 
from xgboost import XGBRegressor, XGBClassifier, plot_importance, plot_tree
#Importing other useful packages
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import statistics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler


# Load training data
train = pd.read_csv('/kaggle/input/molecular-machine-learning/train.csv')
train.info()


#Get list of object columns

train.select_dtypes(include=['object']).columns


double = train


train = pd.concat([train,double])


#Create X,y values
y = train['T80']

train_dependent = train.drop(['Batch_ID', 'Smiles','T80'], axis = 1, inplace = True)


X = train


#Selecting best features for modeling(Don't change any of this code)
feat_select = SelectKBest(f_classif, k='all')
feat_select.fit_transform(X, y)
feat_pvals = pd.DataFrame({'Feature' : X.columns, 'p_value' : feat_select.pvalues_}).sort_values('p_value') 
feat_pvals[feat_pvals['p_value'] < 0.05]


feat_pvals['Feature'].values


features = ['PrimeState', 'SDOS4.5', 'SDOS2.5', 'O19', 'O1', 'SurfaceCharge',
       'SDOS3.7', 'SDOS2.6', 'TDOS1.5', 'O10', 'TDOS1.6', 'O12',
       'LUMO(eV)', 'HAcceptors', 'SDOS2.7', 'O3', 'TDOS1.7', 'SDOS4.9',
       'O8', 'O17', 'RingCount', 'S1', 'PrimeExcite(eV)', 'SDOS4.6', 'T1',
       'TDOS1.8', 'SDOS4.4', 'PrimeExcite(osc)', 'O18', 'SDOS5.0', 'O9',
       'O6', 'SDOS4.8', 'SDOS2.8', 'TDOS3.0', 'LogP', 'SDOS4.7',
       'TDOS1.9', 'Rg', 'TDOS3.1', 'Mass', 'S2', 'HOMO(eV)', 'O5',
       'SDOS3.6', 'TDOS2.4', 'O11', 'TDOS2.0', 'TDOS3.9', 'O2', 'O7',
       'SDOS5.1', 'TDOS2.3', 'SDOS2.9', 'TDOS2.1', 'NumRotatableBonds',
       'S20', 'T3', 'TDOS4.0', 'S18', 'HOMOm1(eV)', 'SDOS4.3', 'T6',
       'S19', 'SDOS3.8', 'TDOS2.2', 'TDOS2.9', 'TDOS3.2', 'O13', 'T4',
       'T7', 'LUMOp1(eV)', 'TDOS3.8', 'NumHeteroatoms', 'TDOS2.5', 'T5',
       'SDOS5.2', 'SDOS5.4', 'SDOS5.3', 'O14', 'S5', 'T8', 'S6', 'S7',
       'T20', 'TDOS3.4', 'S16', 'TDOS3.3', 'T16', 'S15', 'S17', 'SDOS3.5',
       'SDOS4.2', 'O15', 'T9', 'SDOS4.1', 'SDOS3.0', 'TDOS4.7', 'SDOS4.0',
       'S4', 'O16', 'S8', 'T15', 'SDOS3.9', 'Asphericity', 'TDOS3.5',
       'T17', 'TDOS4.1', 'S3', 'TPSA', 'S9', 'T10', 'S13', 'S12',
       'TDOS2.8', 'T13', 'T19', 'S10', 'T14', 'S14', 'S11', 'TDOS3.7',
       'SDOS3.4', 'SDOS3.1', 'T12', 'TDOS3.6', 'T11', 'SDOS3.3',
       'SDOS3.2', 'T18', 'O20', 'TDOS2.6', 'TDOS4.6', 'TDOS2.7',
       'TDOS4.2', 'T2', 'TDOS4.5', 'TDOS4.3', 'TDOS4.4',
       'ChargeCorrection', 'O4', 'DipoleMoment(Debye)', 'HDonors']
X = train[features]


#Split training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Add Parameters
xgb_model = XGBRegressor(n_estimators = 50, max_depth = 30)

#Fitting data to model
xgb_model.fit(train_X, train_y)

#Making predictions on data
xgb_val_predictions = xgb_model.predict(val_X)

#Getting MAE and Accuracy scores
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

#Printing results
print("Validation MAE for Random Forest Model: {:,.0f}".format(xgb_val_mae))
print('The accuracy of the model is: ', xgb_model.score(val_X, val_y)) 
print('The accuracy of the training model is: ', xgb_model.score(train_X, train_y))


#Create finalized model determined hyperparameters from above cell
xgb_final_model = XGBRegressor(n_estimators = 50, max_depth = 30)
#Fit model to full training data
xgb_final_model.fit(X,y)


#load test data
test = pd.read_csv('/kaggle/input/molecular-machine-learning/test.csv')

#Using the same features/variables as our train X value
X_test = test[features]

#Predicting test data's missing y target value
y_pred = xgb_final_model.predict(X_test)


#Generic submission formatting 

output = pd.DataFrame({'Batch_ID': test['Batch_ID'],
                       'T80': y_pred})
output.to_csv('submission.csv', index=False)

