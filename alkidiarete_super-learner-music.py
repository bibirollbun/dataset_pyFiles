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





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import warnings 
warnings.filterwarnings("ignore") 



train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv")


train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)


train = train[train['Number_of_Ads']<10]

features_to_impute = [
    'Episode_Length_minutes',
    'Guest_Popularity_percentage'
]

for feature in features_to_impute:
    value = train[feature].mean()
    train[feature] = train[feature].fillna(value)
    test[feature] = test[feature].fillna(value)



#Mapping Categorical
day_map = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
           'Thursday': 4, 'Friday': 5, 'Saturday': 6}
time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

def preprocessing(df):
    df['Episode_Title'] = df['Episode_Title'].str.replace('Episode', '', regex=False).astype(int)
    df['Publication_Day'] = df['Publication_Day'].map(day_map)
    df['Publication_Time'] = df['Publication_Time'].map(time_map)
    df['Is_Weekend'] = df['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    df['Is_High_Host_Popularity'] = (df['Host_Popularity_percentage'] > 70).astype(int)
    df['Is_High_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 70).astype(int)
    df['Host_Guest_Popularity_Gap'] = df['Host_Popularity_percentage'] - df['Guest_Popularity_percentage']
    df['Ad_Density'] = df['Number_of_Ads'] / df['Episode_Length_minutes']
    df['Ad_Density'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Is_Long_Episode'] = (df['Episode_Length_minutes'] > 60).astype(int)
    
    if 'Episode_Sentiment' in df.columns:
        df['Episode_Sentiment'] = df['Episode_Sentiment'].map(sentiment_map)

    return df

train = preprocessing(train)
test = preprocessing(test)


le_podcast = LabelEncoder()
le_genre = LabelEncoder()

train['Podcast_Name'] = le_podcast.fit_transform(train['Podcast_Name'])
train['Genre'] = le_genre.fit_transform(train['Genre'])
test['Podcast_Name'] = le_podcast.transform(test['Podcast_Name'])
test['Genre'] = le_genre.transform(test['Genre'])


train.info()


X = train.drop(columns=['Listening_Time_minutes'])
y = train['Listening_Time_minutes']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def get_models():
    return [
        XGBRegressor(tree_method='hist', device='cuda'),
        LGBMRegressor(device_type='gpu', verbose=-1),
        CatBoostRegressor(task_type='GPU', verbose=False),
    ]

def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
        train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
        train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
        meta_y.extend(test_y)
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            fold_yhats.append(yhat.reshape(-1, 1))
        meta_X.append(np.hstack(fold_yhats))
    return np.vstack(meta_X), np.array(meta_y)

def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)

def fit_meta_model(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
        yhat = np.maximum(yhat, 0)
        rmse = np.sqrt(mean_squared_error(y, yhat))
        print(f'{model.__class__.__name__}: RMSE {rmse:.3f}')

def super_learner_predictions(X, models, meta_model):
    meta_X = np.hstack([model.predict(X).reshape(-1, 1) for model in models])
    return meta_model.predict(meta_X)

models = get_models()

meta_X, meta_y = get_out_of_fold_predictions(X_train, y_train, models)
print('Meta Data Shape:', meta_X.shape, meta_y.shape)

fit_base_models(X_train, y_train, models)

meta_model = fit_meta_model(meta_X, meta_y)

evaluate_models(X_val, y_val, models)

yhat = super_learner_predictions(X_val, models, meta_model)
yhat = np.maximum(yhat, 0)  
rmse = np.sqrt(mean_squared_error(y_val, yhat))
print(f'Super Learner: RMSE {rmse:.3f}')


test_pred = super_learner_predictions(test, models, meta_model)


sub = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv")

output = pd.DataFrame({"id": sub.id, "Listening_Time_minutes": test_pred})
output.to_csv('submission_ensemble.csv', index=False)

output.head()

