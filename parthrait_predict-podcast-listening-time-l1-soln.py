import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from category_encoders import TargetEncoder

from tqdm import tqdm
from itertools import combinations

import lightgbm as lgb

import warnings
warnings.simplefilter('ignore')


df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col='id')
df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", index_col='id')
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv", index_col='id')


df_train.head()


df_train.isnull().sum()


df_train.info()


df_train.describe()


def feature_eng(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5, 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11, 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17, 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23, 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29, 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35, 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41, 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5, 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df['Episode_Num'] = df['Episode_Title'].str[8:].astype('category')
    
    df['Genre'] = df['Genre'].replace(genr_dict)
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict)
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict)
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict)
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict)
    
    df['Genre'] = df['Genre'].astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].astype('category')
    df['Publication_Day'] = df['Publication_Day'].astype('category')
    df['Publication_Time'] = df['Publication_Time'].astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].astype('category')
    
    df = df.drop(columns=['Episode_Title'])
    return df


df_train = feature_eng(df_train)
df_test = feature_eng(df_test)


encode_columns = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage', 'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
pair_size = [2, 3, 4]

for r in pair_size:
    for cols in tqdm(list(combinations(encode_columns, r))):
        new_col_name = '_'.join(cols)
        
        df_train[new_col_name] = df_train[list(cols)].astype(str).agg('_'.join, axis=1)
        df_train[new_col_name] = df_train[new_col_name].astype('category')
        
        df_test[new_col_name] = df_test[list(cols)].astype(str).agg('_'.join, axis=1)
        df_test[new_col_name] = df_test[new_col_name].astype('category')


df_train=df_train.sample(frac=1)
X = df_train.drop(columns=['Listening_Time_minutes'])
y = df_train['Listening_Time_minutes']


nFolds=2
cv = KFold(nFolds, random_state=97, shuffle=True)
y_pred = np.zeros(len(df_sample))

for idx_train, idx_valid in cv.split(X, y):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    X_test = df_test[X.columns].copy()
    
    encoded_columns = df_train.columns[11:]
    encoder = TargetEncoder(smoothing=0.3)
    
    X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
    X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
    X_test[encoded_columns] = encoder.transform(X_test[encoded_columns])

    model = lgb.LGBMRegressor(
        n_iter=1000,
        max_depth=-1,
        num_leaves=1024,
        colsample_bytree=0.7,
        learning_rate=0.03,
        objective='l2',
        metric='rmse', 
        verbosity=-1,
        device= "cpu",
        random_state=87
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(100)],
    )
    
    y_pred += model.predict(X_test)


df_sample['Listening_Time_minutes'] = y_pred / nFolds
df_sample.to_csv('submission.csv')
df_sample.head()

