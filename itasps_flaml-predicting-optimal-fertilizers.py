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


%%capture
!pip install flaml


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flaml import AutoML
import warnings

warnings.filterwarnings('ignore')


train_file="/kaggle/input/playground-series-s5e6/train.csv"
org_file="/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv"
test_file="/kaggle/input/playground-series-s5e6/test.csv"
sub_file="/kaggle/input/playground-series-s5e6/sample_submission.csv"
index_col="id"
target='Fertilizer Name'
task='classification'
metric='micro_f1' #'log_loss'
eval_method='cv' #resampling strategy:['auto', 'cv', 'holdout']
#estimator_list=['lgbm', 'rf', 'xgboost', 'extra_tree', 'xgb_limitdepth', 'sgd', 'catboost', 'lrl1']
estimator_list=['lgbm']
ensemble_flag = True
TimeBudget= 1.5 * 60 * 60


train = pd.read_csv(train_file,index_col=index_col)
test = pd.read_csv(test_file, index_col=index_col)
sub = pd.read_csv(sub_file, index_col=index_col)
org = pd.read_csv(org_file)


train.shape,test.shape,org.shape


train=pd.concat([train, org])


train.min()


test.min()


train.max()


# FE: create 'Season' column
def create_new_feature_season(df):
    df['Season'] = 'Unknown'
    df.loc[(df['Temparature'] > 30) & (df['Humidity'] > 60), 'Season'] = 'Summer'
    df.loc[(df['Temparature'] < 20) & (df['Humidity'] > 70), 'Season'] = 'Winter'
    df.loc[(df['Temparature'] >= 20) & (df['Temparature'] <= 30) & (df['Humidity'] >= 40) & (df['Humidity'] <= 70), 'Season'] = 'Spring'
    df.loc[(df['Temparature'] >= 20) & (df['Temparature'] <= 30) & (df['Humidity'] >= 70), 'Season'] = 'Rainy'
    return df

"""
def create_new_features(df):
    #df["Soil_Crop"] = df['Soil Type'] + df['Crop Type']
    #df["NPh"] =  df['Phosphorous']/df['Nitrogen']
    #df["NPo"] =  df['Potassium'] /df['Nitrogen']
    df["NK"] =  df['Nitrogen'] - df['Potassium']
    df["KP"] =  df['Potassium'] - df['Phosphorous']
    return df"""

train = create_new_feature_season(train)
test = create_new_feature_season(test)

#train = create_new_features(train)
#test = create_new_features(test)


features=list(test)
features.append(target)
train=train[features]


train.sample(2)


test.sample(2)


train.info()


test.info()


train[target].value_counts()


aml = AutoML()

aml.fit(train.drop(columns=target,axis=1), train[target],
        task=task, metric=metric,time_budget= TimeBudget ,eval_method=eval_method, 
        n_jobs = -1, seed=47,ensemble = ensemble_flag) 

        #keep_search_state= True, split_type= 'group',
        #groups = train.Season.values,
        #n_splits = 3,


preds = aml.predict_proba(test)
preds


"""
if ensemble_flag == False:
    print(f"{aml.best_estimator=}")
    print(f"{aml.best_config=}")
    print(f"params for best estimator: {aml.model.config2params(aml.best_config)}")
"""
#StackingClassifier' object has no attribute 'config2params'


# best loss per estimator
pd.Series(aml.best_loss_per_estimator).sort_values()


# save model
import pickle
with open('automl_emissions.pkl', 'wb') as f:
    pickle.dump(aml, f, pickle.HIGHEST_PROTOCOL)


"""
if ensemble_flag == False:
    feat_imp = pd.Series(
        aml.model.feature_importances_,
        train.drop(columns=target,axis=1).columns
    ).sort_values(ascending=True)
    
    fig,ax = plt.subplots(1,1,figsize=(6,4))
    feat_imp.plot(kind='barh',ax=ax)
    _ = ax.set_title(f'Features importances from {aml.best_estimator}')
"""
#'StackingClassifier' object has no attribute 'feature_importances_'


aml.best_config_per_estimator


df = pd.DataFrame(preds,columns=aml.classes_.tolist())
df.head()


df.to_csv('best_automl_prob.csv')


df[target] = df.apply(lambda x: ' '.join(x.sort_values(ascending=False).index[:3]), axis=1)
df.head()


sub[target]=list(df[target] )
sub.to_csv('submission.csv')


sub

