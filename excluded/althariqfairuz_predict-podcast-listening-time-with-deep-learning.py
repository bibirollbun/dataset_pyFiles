!pip install autogluon
!pip install -U ipywidgets


import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore')

from autogluon.tabular import TabularDataset, TabularPredictor


train_df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv' , index_col = 'id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col = 'id')


train_df.info()


train_df.head()


train_df.isnull().sum() / len(train_df) * 100


test_df.info()


test_df.head()


test_df.isnull().sum() / len(train_df) * 100


for col in train_df.select_dtypes(include=['object']).columns:
    print(f'Column name: {col}\nTotal unqiue values: {train_df[col].nunique()}\n')


for col in test_df.select_dtypes(include=['object']).columns:
    print(f'Column name: {col}\nTotal unqiue values: {test_df[col].nunique()}\n')


label = 'Listening_Time_minutes'


!nvidia-smi


predictor = TabularPredictor(label = label, eval_metric='rmse', problem_type='regression').fit(train_df, 
                                                                                               presets='medium_quality', 
                                                                                               time_limit=3600*9,
                                                                                               verbosity=3,
                                                                                               ag_args_fit={'num_gpus': 2}
                                                                                              )
results = predictor.fit_summary()


predictor.leaderboard()


df = predictor.predict(test_df).to_frame(name=label)
df.head()


df.to_csv('autogluon_medium_quality_gpu2.csv')

