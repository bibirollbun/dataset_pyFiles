import numpy as np 
import pandas as pd 

import os

import h2o
from h2o.automl import H2OAutoML

import matplotlib as plt
%matplotlib inline
        
import warnings
warnings.filterwarnings("ignore")        
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import seaborn as sns
import squarify


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')

RMV = ["id","Listening_Time_minutes","Episode_Length_minutes"]
TARGET = ['Listening_Time_minutes']
FEATURES = [c for c in train.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}\n")

CATS = []
for c in FEATURES:
    if train[c].dtype=="object":
        CATS.append(c)
        train[c] = train[c].fillna("NAN")
        test[c] = test[c].fillna("NAN")
print(f"In these features, there are {len(CATS)} CATEGORICAL FEATURES: {CATS}")

display(train.sample(8))
display(train.info())


display(train.query('Listening_Time_minutes > Episode_Length_minutes'))
train.loc[train.query('Listening_Time_minutes > Episode_Length_minutes').index , 'Listening_Time_minutes']= train['Episode_Length_minutes']


display(train.query('Number_of_Ads > 10').index)
train = train.drop(train.query('Number_of_Ads > 5').index)


train['Episode'] = train['Episode_Title'].str.split(' ').str[1]
test['Episode'] = test['Episode_Title'].str.split(' ').str[1]

train['Genre_Podcast_Name'] = train['Genre'] + train['Podcast_Name']
test['Genre_Podcast_Name'] = test['Genre'] + test['Podcast_Name']

train['Day_Time'] = train['Publication_Day'] + train['Publication_Time']
test['Day_Time'] = test['Publication_Day'] + test['Publication_Time']

train['BIN_Hpp']=0
train.loc[train['Host_Popularity_percentage']<=20,'BIN_Hpp']=1
train.loc[(train['Host_Popularity_percentage']>20)&(train['Host_Popularity_percentage']<=40),'BIN_Hpp']=2
train.loc[(train['Host_Popularity_percentage']>40)&(train['Host_Popularity_percentage']<=80),'BIN_Hpp']=3
train.loc[train['Host_Popularity_percentage']>80,'BIN_Hpp']=4

test['BIN_Hpp']=0
test.loc[test['Host_Popularity_percentage']<=20,'BIN_Hpp']=1
test.loc[(test['Host_Popularity_percentage']>20)&(test['Host_Popularity_percentage']<=40),'BIN_Hpp']=2
test.loc[(test['Host_Popularity_percentage']>40)&(test['Host_Popularity_percentage']<=80),'BIN_Hpp']=3
test.loc[test['Host_Popularity_percentage']>80,'BIN_Hpp']=4


SEED = 0
h2o.init()


#h2o.remove(aml)   if you want to try again

train_hframe = h2o.H2OFrame(train.drop(columns=['id','Episode_Title']))


splits = train_hframe.split_frame(ratios=[0.9], seed=SEED)
train_ = splits[0]
valid_ = splits[1]
print("train: %d test: %d" % (train_.nrows, valid_.nrows))

features = train_.columns
features.remove('Listening_Time_minutes')  

aml = H2OAutoML( max_runtime_secs = 5000, project_name='regression', sort_metric= 'rmse', stopping_metric = 'rmse',seed=SEED,
                exclude_algos = ["GLM", "DeepLearning", "DRF"])
aml.train(x=features, y='Listening_Time_minutes', training_frame=train_, validation_frame=valid_)

test_ = h2o.H2OFrame(test.drop(columns=['id','Episode_Title']))
pred = aml.predict(test_)


aml.leaderboard   
 


model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_All" in mid][0])
model = h2o.get_model([mid for mid in model_ids if "_model_" in mid][0])



se


model


#aml.leader.model_performance(test_data=valid_)


test_hframe = h2o.H2OFrame(test.drop(columns=['id']))
pred = aml.predict(test_)


shap_plot = model.shap_summary_plot(valid_)


#ra_plot = se.residual_analysis_plot(valid_)



ra_plot = model.shap_explain_row_plot(valid_, row_index=0)


learning_curve_plot = model.learning_curve_plot()


va_plot = h2o.varimp_heatmap(aml.leaderboard.sort("rmse"))


aml.leader.explain(valid_[:100],exclude_explanations =["pdp", "ice"])


pred_df = pred.as_data_frame(use_multi_thread=True)
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission['Listening_Time_minutes'] = pred_df['predict']
submission.to_csv(f"submission.csv",index=False)
submission


h2o.cluster().shutdown()

