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


import polars as pl

from tqdm import tqdm
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
import lightgbm as lgb


PODCAST_MAP = {name: idx for idx, name in enumerate([
    'Mystery Matters', 'Joke Junction', 'Study Sessions', 'Digital Digest', 'Mind & Body',
    'Fitness First', 'Criminal Minds', 'News Roundup', 'Daily Digest', 'Music Matters',
    'Sports Central', 'Melody Mix', 'Game Day', 'Gadget Geek', 'Global News', 'Tech Talks',
    'Sport Spot', 'Funny Folks', 'Sports Weekly', 'Business Briefs', 'Tech Trends', 'Innovators',
    'Health Hour', 'Comedy Corner', 'Sound Waves', 'Brain Boost', "Athlete's Arena", 'Wellness Wave',
    'Style Guide', 'World Watch', 'Humor Hub', 'Money Matters', 'Healthy Living', 'Home & Living',
    'Educational Nuggets', 'Market Masters', 'Learning Lab', 'Lifestyle Lounge', 'Crime Chronicles',
    'Detective Diaries', 'Life Lessons', 'Current Affairs', 'Finance Focus', 'Laugh Line',
    'True Crime Stories', 'Business Insights', 'Fashion Forward', 'Tune Time'
])}
GENRE_MAP = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4,
             'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
DAY_MAP = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
           'Friday': 4, 'Saturday': 5, 'Sunday': 6}
TIME_MAP = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
SENTIMENT_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

ENCODE_COLUMNS = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage',
                  'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
PAIR_SIZES = [2, 3, 4]


def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', index_col='id')
    test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', index_col='id')
    sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')
    return train, test, sample_sub


def preprocess(df):
    df['Episode_Num'] = df['Episode_Title'].str.extract(r'(\d+)$')[0].astype('category')
    
    df['Podcast_Name'] = df['Podcast_Name'].map(PODCAST_MAP).astype('category')
    df['Genre'] = df['Genre'].map(GENRE_MAP).astype('category')
    df['Publication_Day'] = df['Publication_Day'].map(DAY_MAP).astype('category')
    df['Publication_Time'] = df['Publication_Time'].map(TIME_MAP).astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].map(SENTIMENT_MAP).astype('category')

    df.drop(columns=['Episode_Title'], inplace=True)
    return df


def create_pair_features(df_train, df_test):
    for r in PAIR_SIZES:
        for cols in tqdm(list(combinations(ENCODE_COLUMNS, r))):
            new_col = '_'.join(cols)
            for df in [df_train, df_test]:
                df[new_col] = df[list(cols)].astype(str).agg('_'.join, axis=1).astype('category')



def train_and_predict(train, test, sample_sub):
    X = train.drop(columns=['Listening_Time_minutes'])
    y = train['Listening_Time_minutes']
    y_pred = np.zeros(len(test))

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Categorical features are encoded after feature interaction
    cat_features = train.columns.difference(['Listening_Time_minutes'])

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        X_test = test[X.columns].copy()

        encoder = TargetEncoder()
        X_train[cat_features] = encoder.fit_transform(X_train[cat_features], y_train)
        X_val[cat_features] = encoder.transform(X_val[cat_features])
        X_test[cat_features] = encoder.transform(X_test[cat_features])

        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=1024,
            colsample_bytree=0.7,
            max_depth=-1,
            max_bin=255,
            objective='l2',
            metric='rmse',
            verbosity=-1,
            device='gpu',
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(100)])

        y_pred += model.predict(X_test)

    sample_sub['Listening_Time_minutes'] = y_pred / cv.get_n_splits()
    sample_sub.to_csv('submission.csv')
    return sample_sub


if __name__ == '__main__':
    train_df, test_df, submission_df = load_data()
    train_df = preprocess(train_df)
    test_df = preprocess(test_df)
    create_pair_features(train_df, test_df)
    final_submission = train_and_predict(train_df, test_df, submission_df)
    print(final_submission.head())




