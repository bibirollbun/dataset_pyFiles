!pip install -qq scikit-learn==1.6.1


import pandas as pd
import warnings
from itertools import combinations
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor
import numpy as np
import lightgbm as lgb
import sklearn

warnings.simplefilter(action = 'ignore', category = pd.errors.SettingWithCopyWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

podcast = pd.read_csv('/kaggle/input/podcast-new-features/podcast_new_features.csv')
podcast_test = pd.read_csv('/kaggle/input/podcast-test-new-features/podcast_test_new_features.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

podcast = podcast[~podcast.Listening_Time_minutes.isna()].reset_index(drop = True)
x = podcast.drop(columns = ['Listening_Time_minutes','Unnamed: 0','id'])
y = podcast.Listening_Time_minutes

podcast_test = podcast_test.drop(columns = ['Unnamed: 0', 'id'])

encoder = TargetEncoder(random_state = 42)
x = encoder.fit_transform(x,y)
podcast_test = encoder.transform(podcast_test)

model = lgb.LGBMRegressor(
        n_iter=1000,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        max_bin=1024,
    )

model.fit(x,y)
predictions = model.predict(podcast_test)
submission.Listening_Time_minutes = predictions
submission.to_csv('submission.csv', index = False)

