%%time

import pandas as pd 
import numpy as np
!pip install -qq lifelines

%config InlineBackend.figure_format = 'retina'


%%time

SEED = 42

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase

train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
sample = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

cat_c = ['Episode_Title', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment','Podcast_Name','Genre']

def update(df):

    for col in cat_c:
        df[col] = df[col].astype('category')
    return df

train = update(train)
test = update(test)

train.head()


%%time

print(f'Train Data Shape: {train.shape}')
print(f'Test Data Shape: {test.shape}')

print(f'\nTrain Data Duplicated Values: {train.duplicated().sum()}')
print(f'Test Data Duplicated Values: {test.duplicated().sum()}')


%%time

encode_c = {'cat_c': cat_c}

base = AbdBase(train_data=train, test_data=test, target_column='Listening_Time_minutes',gpu=True,
                 problem_type="regression", metric="rmse", seed=SEED,ohe_fe=False,ordinal_encoder=encode_c,
                 n_splits=5,early_stop=True,num_classes=0,cat_features=False,
                 fold_type='KF')


%%time

base.X_train.head()


%%time

# 13.0229 No FE

ParamsCat = {'n_estimators': 2000, 'max_depth': 8, 'learning_rate': 0.12383646629066654, 
             'min_child_samples': 60, 'eval_metric':'RMSE'}

results_CAT_1 = base.Train_ML(ParamsCat,'CAT',e_stop=50) # LB : 13.29935 || CV : 13.2180


%%time

mp = results_CAT_1[1]

sample['Listening_Time_minutes'] = mp
sample.to_csv('submission.csv', index=False)
sample.head()

