import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
from sklearn.model_selection import cross_val_score, cross_val_predict,train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import  Ridge,BayesianRidge,LassoLars
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from sklearn.preprocessing import KBinsDiscretizer

from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import  make_scorer, mean_squared_error
from random import sample

from category_encoders import *

import warnings
warnings.filterwarnings('ignore')


def scorer(target,pred):
    return mean_squared_error(target,pred,squared=False)


myscorer=make_scorer(mean_squared_error,greater_is_better=False,squared=False)
nfolds=10


def pickle_var(path,var=None):
    import pickle
    '''
    wrapper for pickle
    path is a path to a file
    if var is None is supposed you want load a pickled file so it returns
     pickle load
    else is supposed you want dump var to file
    '''
    #complex types as np.array or pandas DataFrame don't allow a simple comparison with None
    if type(var)==type(None):
         with open(path,'rb') as fich:
            return pickle.load(fich)

    else:
        with open(path,'wb') as fich:
            pickle.dump(var,fich)




root_path=Path('/kaggle')
input_path=root_path/'input/playground-series-s5e2'
output_path=root_path/'working'
temp_path=root_path/'temp'

for file in input_path.glob('*.csv'):
    print(file)


dataset=pd.read_csv(input_path/'train.csv',index_col='id')
datasetplus=pd.read_csv(input_path/'training_extra.csv',index_col='id')
dataset=pd.concat([dataset,datasetplus])
testset=pd.read_csv(input_path/'test.csv',index_col='id')


target_col='Price'
cat_cols=['Brand', 'Material', 'Size','Laptop Compartment', 'Waterproof', 'Style', 'Color']
num_cols=['Weight Capacity (kg)']


target=dataset[target_col]
data=dataset.drop(target_col,axis=1)


encoder=SummaryEncoder(cols=cat_cols,quantiles=[0.25,0.5,0.75],m=1.0,handle_missing='return_nan')


data=encoder.fit_transform(data,target)
testset=encoder.transform(testset)


from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=10, random_state=0)


data[data.columns]=imp.fit_transform(data)
testset[testset.columns]=imp.transform(testset)


model=make_pipeline(StandardScaler(),Ridge())
model.fit(data,target)


submission=pd.read_csv(input_path/'sample_submission.csv',index_col='id')


submission[target_col]=model.predict(testset)


if submission[target_col].isnull().sum():
    submission=submission.fillna(target.median())


submission.to_csv('submission.csv')


submission




