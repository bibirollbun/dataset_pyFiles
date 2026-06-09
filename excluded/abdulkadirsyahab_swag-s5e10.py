# import library
import numpy as np
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# read dataset
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


# view train dataset
train.head()


# view test dataset
test.head()


# view sample_submission dataset
sample_submission.head()


# view train info
train.info()


# view test info
test.info()


# added some feature engineering
train['road_capacity'] = train['num_lanes'] * train['speed_limit']
test['road_capacity'] = test['num_lanes'] * test['speed_limit']

train['curvature_risk'] = train['curvature'] * train['speed_limit']
test['curvature_risk'] = test['curvature'] * test['speed_limit']

train['accident_rate'] = train['num_reported_accidents']/train['road_capacity']
test['accident_rate'] = test['num_reported_accidents']/test['road_capacity']


# import library to encoding & transform
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# encode & transofrm data
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
binary_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('bin', 'passthrough', binary_cols)
    ]
)


# set train & test data
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

x = train.drop(['accident_risk'], axis=1)
y = train['accident_risk']


# import pipeline & model
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# make model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])

# fit model
model.fit(x, y)


# prediction
prediction = model.predict(test)


# save model to sample_sumbission
sample_submission['accident_risk'] = prediction
sample_submission.to_csv('sample_submission.csv', index=False)

# see the sample_submission head
sample_submission.head()




