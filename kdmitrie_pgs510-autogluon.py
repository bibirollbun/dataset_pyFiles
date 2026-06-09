!pip install autogluon.tabular

from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import pandas as pd


df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e10/train.csv').drop(columns = 'id')
df_test  = pd.read_csv(r'/kaggle/input/playground-series-s5e10/test.csv').drop(columns = 'id')


train = TabularDataset(df_train)
test = TabularDataset(df_test)

automl = TabularPredictor(label='accident_risk', problem_type='regression', eval_metric='root_mean_squared_error')
#automl.fit(train, presets='best_quality', time_limit=120)
automl.fit(train, presets='best_quality')


automl.leaderboard()


prediction = automl.predict(test)

data_submit = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
data_submit.accident_risk = prediction
data_submit[['id', 'accident_risk']].to_csv('submission.csv', index=False)


!head submission.csv

