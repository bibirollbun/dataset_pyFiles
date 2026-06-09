import random
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import warnings
warnings.filterwarnings(action='ignore', category=RuntimeWarning)


PATH_test = '/kaggle/input/home-credit-default-risk/application_test.csv'
PATH_train = '/kaggle/input/home-credit-default-risk/application_train.csv'

df_test = pd.read_csv(PATH_test)
df = pd.read_csv(PATH_train)


seed = 1723

random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


df.dropna(inplace=True)


cats = list(df.select_dtypes(include=['object']).columns)
nums = [c for c in df.columns if c not in cats+['TARGET', 'SK_ID_CURR']]

df[cats] = df[cats].astype('category')


df, test = train_test_split(df, 
                            test_size=.20, 
                            random_state=1723)
df.reset_index(drop=True, 
               inplace=True)
test.reset_index(drop=True, 
                 inplace=True)

inds = []
kf = KFold(n_splits=5, 
           shuffle=True, 
           random_state=1723)
for (train_index, valid_index) in kf.split(df):
    inds += [[train_index, valid_index]]


def training_phase(df):

    oof_ms = []
    test_esti = np.zeros(test.shape[0])
    
    for fold in range(len(inds)):

        train_index = inds[fold][0]
        valid_index = inds[fold][1]
        
        train = df.iloc[train_index]
        train.reset_index(drop=True, 
                          inplace=True)
        
        valid = df.iloc[valid_index]
        valid.reset_index(drop=True, 
                          inplace=True)
        
        train_dataset = lgb.Dataset(train[nums+cats], 
                                    train['TARGET'], 
                                    categorical_feature=cats)
        valid_dataset = lgb.Dataset(valid[nums+cats], 
                                    valid['TARGET'], 
                                    categorical_feature=cats)
    
        params = {
            "objective": "binary", 
            "boosting_type": "gbdt", 
            "num_boost_rounds": 1000, 
            "early_stopping_round": 100, 
            "max_depth": -1, 
            "bagging_fraction": .5, 
            "features_fraction": .5, 
            "verbose": -1, 
            "nthreads": 4, 
            "random_state": 1723
        }
        
        model = lgb.train(params=params, 
                          train_set=train_dataset, 
                          valid_sets=valid_dataset)
    
        valid_esti = model.predict(valid[nums+cats])
        oof_ms += [roc_auc_score(valid['TARGET'], 
                                 valid_esti)*2-1]
    
        test_esti += model.predict(test[nums+cats])
        if fold == 4:
            print (oof_ms, 
                   roc_auc_score(test['TARGET'], 
                                 test_esti)*2-1)
        

training_phase(df)

