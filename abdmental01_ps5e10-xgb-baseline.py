%%time

import pandas as pd 
import numpy as np


%%time

SEED = 42

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase

PATH = "/kaggle/input/playground-series-s5e10"

train = pd.read_csv(f'{PATH}/train.csv')
sample = pd.read_csv(f'{PATH}/sample_submission.csv')
test = pd.read_csv(f'{PATH}/test.csv')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

cat_c = ['road_type', 'lighting', 'weather', 'time_of_day']

def update(df):

    for col in cat_c:
        df[col] = df[col].astype('category')

    return df

train = update(train)
test = update(test)

train.head()


%%time

cat_c = ['road_type', 'lighting', 'weather', 'time_of_day']

ohe_fe = {'cat_c': cat_c}

base = AbdBase(train_data=train, test_data=test, target_column='accident_risk',gpu=True,
                 problem_type="regression", metric="rmse", seed=SEED,ohe_fe = False,
                 n_splits=5,early_stop=True,num_classes=0,cat_features=False,
                 fold_type='RKF')


%%time

base.X_train.head()


%%time

ParamsXgb = {
    'n_estimators': 20000, 'max_depth': 7, 'learning_rate': 0.02773314923650538, 'min_child_weight': 7, 
    'subsample': 0.6971184489228035, 'enable_categorical':True}

results_XGB_1 = base.Train_ML(ParamsXgb,'XGB',e_stop=50)


%%time

mp = results_XGB_1[1]

sample['accident_risk'] = mp
sample.to_csv('submission.csv', index=False)
sample.head()

