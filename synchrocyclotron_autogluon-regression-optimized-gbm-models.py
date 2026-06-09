import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


%pip install autogluon.tabular


import numpy as np
import pandas as pd


from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common import space


data = TabularDataset("/kaggle/input/playground-series-s5e2/train.csv")
test = TabularDataset("/kaggle/input/playground-series-s5e2/test.csv")


data.head()


id_column = 'id'
target = 'Price'


train_data = data.drop(columns=[id_column])
test_ids = test[id_column]
test_data = test.drop(columns=[id_column])


hyperparameters = {
    'GBM': {},  # LightGBM with default settings
    'CAT': {},  # CatBoost with default settings
    'XGB': {}   # XGBoost with default settings
}


predictor = TabularPredictor(label=target, eval_metric='root_mean_squared_error').fit(
    train_data, 
    presets='best_quality', 
    time_limit=3600,
    verbosity=4)


predictions = predictor.predict(test_data)


submission = pd.DataFrame({id_column: test_ids, target: predictions})
submission.to_csv("submission.csv", index=False)


print(submission.head())


predictor.leaderboard()

