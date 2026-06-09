!pip install -q autogluon.tabular
!pip install -q ray
!pip install -U ipywidgets


from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import numpy as np
import warnings
import shutil

warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv",index_col='id')
train.head()


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv",index_col='id')
test.head()


train = TabularDataset(train)
test = TabularDataset(test)


TARGET = 'Price'
TIME_LIMIT = 3600 * 10


predictor = TabularPredictor(
    label=TARGET,
    eval_metric = 'rmse',                        
    problem_type = "regression",
    verbosity=3
)


predictor.fit(train,
              presets = 'best_quality',    
              excluded_model_types = ['KNN'],
              time_limit = TIME_LIMIT
             )


predictor.leaderboard()


test_preds = predictor.predict(test)
test_preds


sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
sub[TARGET] = test_preds.values
sub.to_csv('submission_ag.csv', index=False)
sub.head()

