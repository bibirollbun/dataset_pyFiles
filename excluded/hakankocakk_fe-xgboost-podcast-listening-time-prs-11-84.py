!pip install -qq scikit-learn==1.6.1


import numpy as np
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from itertools import combinations

from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

import xgboost as xgb
from xgboost import XGBRegressor

import warnings
warnings.simplefilter('ignore')


def data_collection():
    df_train = pd.read_csv("/kaggle/input/playground-series-s5e4/train.csv", index_col='id')
    df_test = pd.read_csv("/kaggle/input/playground-series-s5e4/test.csv", index_col='id')
    df_subm = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv', index_col='id')
    
    return df_train, df_test, df_subm

df_train, df_test, df_subm = data_collection()


def feature_engineering(df):
    podc_dict = {'Mystery Matters': 0, 'Joke Junction': 1, 'Study Sessions': 2, 'Digital Digest': 3, 'Mind & Body': 4, 'Fitness First': 5,
                 'Criminal Minds': 6, 'News Roundup': 7, 'Daily Digest': 8, 'Music Matters': 9, 'Sports Central': 10, 'Melody Mix': 11,
                 'Game Day': 12, 'Gadget Geek': 13, 'Global News': 14, 'Tech Talks': 15, 'Sport Spot': 16, 'Funny Folks': 17,
                 'Sports Weekly': 18, 'Business Briefs': 19, 'Tech Trends': 20, 'Innovators': 21, 'Health Hour': 22, 'Comedy Corner': 23,
                 'Sound Waves': 24, 'Brain Boost': 25, "Athlete's Arena": 26, 'Wellness Wave': 27, 'Style Guide': 28, 'World Watch': 29,
                 'Humor Hub': 30, 'Money Matters': 31, 'Healthy Living': 32, 'Home & Living': 33, 'Educational Nuggets': 34, 'Market Masters': 35,
                 'Learning Lab': 36, 'Lifestyle Lounge': 37, 'Crime Chronicles': 38, 'Detective Diaries': 39, 'Life Lessons': 40, 'Current Affairs': 41,
                 'Finance Focus': 42, 'Laugh Line': 43, 'True Crime Stories': 44, 'Business Insights': 45, 'Fashion Forward': 46, 'Tune Time': 47}
    
    genr_dict = {'True Crime': 0, 'Comedy': 1, 'Education': 2, 'Technology': 3, 'Health': 4, 'News': 5,
                 'Music': 6, 'Sports': 7, 'Business': 8, 'Lifestyle': 9}
    
    week_dict = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    time_dict = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    sent_dict = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    df["Episode_Num"] = df["Episode_Title"].str.replace("Episode ", "").astype('category')
    
    df['Genre'] = df['Genre'].replace(genr_dict).astype('category')
    df['Podcast_Name'] = df['Podcast_Name'].replace(podc_dict).astype('category')
    df['Publication_Day'] = df['Publication_Day'].replace(week_dict).astype('category')
    df['Publication_Time'] = df['Publication_Time'].replace(time_dict).astype('category')
    df['Episode_Sentiment'] = df['Episode_Sentiment'].replace(sent_dict).astype('category')
    
    df = df.drop(columns=['Episode_Title'])
    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)


def feature_engineering_2(df):
    encode_columns = ['Episode_Length_minutes', 'Episode_Num', 'Host_Popularity_percentage',
                      'Number_of_Ads', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time',
                      'Genre', 'Guest_Popularity_percentage']
    pair_size = [2, 3, 4]
    
    for r in pair_size:
        for cols in tqdm(list(combinations(encode_columns, r))):
            new_col_name = '_'.join(cols)
            
            df[new_col_name] = df[list(cols)].astype(str).agg('_'.join, axis=1)
            df[new_col_name] = df[new_col_name].astype('category')

    return df

df_train = feature_engineering_2(df_train)
df_test = feature_engineering_2(df_test)


def split(df):
    X = df_train.drop(columns=['Listening_Time_minutes'])
    y = df_train['Listening_Time_minutes']

    return X, y

X, y = split(df_train)


def XGBoost_Model(X, y):
    cv = KFold(5, random_state=42, shuffle=True)
    y_pred = np.zeros(len(df_subm))
    
    for idx_train, idx_valid in cv.split(X, y):
        X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
        X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
        X_test = df_test[X.columns].copy()
        
        encoded_columns = df_train.columns[11:]
        encoder = TargetEncoder(random_state=42)
        
        X_train[encoded_columns] = encoder.fit_transform(X_train[encoded_columns], y_train)
        X_valid[encoded_columns] = encoder.transform(X_valid[encoded_columns])
        X_test[encoded_columns] = encoder.transform(X_test[encoded_columns])

        model = XGBRegressor(
            tree_method='hist',
            device='cuda',
            max_depth=14,
            colsample_bytree=0.5,
            subsample=0.9,
            n_estimators=10000,
            learning_rate=0.02,
            enable_categorical=True,
            min_child_weight=10,
            early_stopping_rounds=150,
        )
        
    
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=100,
        )

        
        
        y_pred += model.predict(X_test)

    return y_pred / 5


y_pred = XGBoost_Model(X, y)


def submission(predict, model_name):
    submission_df = pd.read_csv("/kaggle/input/playground-series-s5e4/sample_submission.csv",encoding='utf-8',low_memory=False)
    submission_df["Listening_Time_minutes"] = predict
    submission_df.to_csv(f'/kaggle/working/{model_name}_submission.csv', index=False)

submission(y_pred, "XGBoostV3")




