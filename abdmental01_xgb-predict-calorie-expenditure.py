%%time

import pandas as pd 
import numpy as np
!pip install -qq lifelines


%%time

SEED = 42

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase

train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

cat_c = ['Sex']

def update(df):

    for col in cat_c:
        df[col] = df[col].astype('category')
    return df

train = update(train)
test = update(test)

train.drop_duplicates(inplace=True)

train.head()


%%time

print(f'Train Data Shape: {train.shape}')
print(f'Test Data Shape: {test.shape}')

print(f'\nTrain Data Duplicated Values: {train.duplicated().sum()}')
print(f'Test Data Duplicated Values: {test.duplicated().sum()}')


%%time

def N_FE(df):

    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Exercise_Intensity'] = df['Heart_Rate'] / df['Duration']
    df['Temp_Deviation'] = df['Body_Temp'] - 37.0

    return df


train = N_FE(train)
test = N_FE(test)


%%time

train.head()


%%time

cat_c = ['Sex']

encode_c = {'cat_c': cat_c}

base = AbdBase(train_data=train, test_data=test, target_column='Calories',gpu=True,
                 problem_type="regression", metric="rmsle", seed=SEED,ohe_fe=encode_c,ordinal_encoder=False,
                 n_splits=5,early_stop=True,num_classes=0,cat_features=False,
                 fold_type='KF')


%%time

base.X_train.head()


%%time

ParamsXGB = {'n_estimators': 10000, 'max_depth': 10, 'learning_rate': 0.014429896155403753, 'min_child_weight': 9,
             'subsample': 0.773872391545965, 'colsample_bytree': 0.8161189090550527, 'gamma': 1.7680328822531155,
             'reg_alpha': 0.23689913288008635, 'reg_lambda': 2.525898164980985}

results_XGB_1 = base.Train_ML(ParamsXGB,'XGB',e_stop=50) 


%%time

mp = results_XGB_1[1]

sample['Calories'] = mp
sample.to_csv('submission.csv', index=False)
sample.head()

