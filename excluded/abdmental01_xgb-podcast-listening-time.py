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


%%time

def n_fe(df):
    import numpy as np
    
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)
    
    return df

train = n_fe(train)
test = n_fe(test)


%%time

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

ParamsXGB = {'max_depth': 10, 'learning_rate': 0.00505052132642901, 'min_child_weight': 9, 
             'subsample': 0.8261771678236348, 'colsample_bytree': 0.7657134251473725, "n_estimators": 10000, 
             'gamma': 1.6081028372742674, 'reg_alpha': 0.11243311001772383, 'reg_lambda': 0.15742461454226775}

results_XGB_1 = base.Train_ML(ParamsXGB,'XGB',e_stop=300) # 12.6963


%%time

mp = results_XGB_1[1]

sample['Listening_Time_minutes'] = mp
sample.to_csv('submission.csv', index=False)
sample.head()

