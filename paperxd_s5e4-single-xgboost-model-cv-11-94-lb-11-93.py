!pip install -qq scikit-learn==1.6.1


import pandas as pd
from itertools import combinations
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBRegressor
import numpy as np

podcast = pd.read_csv('/kaggle/input/podcast-new-features/podcast_new_features.csv')
podcast_test = pd.read_csv('/kaggle/input/podcast-test-new-features/podcast_test_new_features.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

podcast = podcast[~podcast.Listening_Time_minutes.isna()].reset_index(drop = True)
x = podcast.drop(columns = ['Listening_Time_minutes','Unnamed: 0','id'])
y = podcast.Listening_Time_minutes

org_columns = ['Episode_Length_minutes','Host_Popularity_percentage','Guest_Popularity_percentage']
cat_columns = ['Host_Popularity_percentage','Guest_Popularity_percentage']
podcast_test = podcast_test.drop(columns = ['Unnamed: 0', 'id'])

x_enc = x.drop(columns = org_columns)
x_org = x[org_columns]

podcast_test_enc = podcast_test.drop(columns = org_columns)
podcast_test_org = podcast_test[org_columns]

for column in cat_columns:
    x_org[column] = x_org[column].astype('category')
    podcast_test_org[column] = podcast_test_org[column].astype('category')
encoder = TargetEncoder(random_state = 42)
x_enc = pd.DataFrame(encoder.fit_transform(x_enc,y), columns = x_enc.columns)
podcast_test_enc = pd.DataFrame(encoder.transform(podcast_test_enc), columns = podcast_test_enc.columns)

x = pd.concat([x_enc, x_org], axis = 1)
podcast_test = pd.concat([podcast_test_enc, podcast_test_org], axis = 1)

params = {'n_estimators': 1000, 'max_depth': 14, 'learning_rate': 0.011015629090199364, 'subsample': 0.9116787269528593, 'colsample_bytree': 0.7827627872771932, 'colsample_bylevel': 0.6648599558828336, 'min_child_weight': 11, 'gamma': 5.2709692650185715, 'reg_lambda': 0.012221079454198924, 'reg_alpha': 8.45962277411465}
model = XGBRegressor(**params, enable_categorical = True, random_state = 42, verbosity = 2)
model.fit(x,y)
predictions = model.predict(podcast_test)
submission.Listening_Time_minutes = predictions
submission.to_csv('submission.csv', index = False)

