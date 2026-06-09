# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as r
import sklearn
import seaborn as sns
import category_encoders as ce
import matplotlib.pyplot as plt
import wordcloud
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


def engineer_features(X_train, X_test):
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

    # 1. Ad Density
    combined['ads_per_minute'] = combined['Number_of_Ads'] / (combined['Episode_Length_minutes'] + 1e-3)

    # 2. Is Weekend
    combined['is_weekend'] = combined['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)

    # 3. Time of Day Features
    combined['is_morning'] = (combined['Publication_Time'] == 'Morning').astype(int)
    combined['is_night'] = (combined['Publication_Time'] == 'Night').astype(int)

    # 4. Episode Length Buckets
    combined['length_bucket'] = pd.cut(combined['Episode_Length_minutes'], bins=[0, 30, 60, 90, 200],
                                       labels=['short', 'medium', 'long', 'very_long'])

    # 5. Sentiment Ordinal Mapping
    sentiment_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    combined['sentiment_score'] = combined['Episode_Sentiment'].map(sentiment_map)

    # 6. Host-Guest Popularity Ratio
    combined['popularity_ratio'] = combined['Guest_Popularity_percentage'] / (
        combined['Host_Popularity_percentage'] + 1e-3)

    # 7. Episode Number from Title
    combined['episode_number'] = combined['Episode_Title'].str.extract(r'(\d+)').astype(float)

    # 8. Genre + Sentiment Interaction
    combined['genre_sentiment'] = combined['Genre'].astype(str) + "_" + combined['Episode_Sentiment'].astype(str)

    # --- Handle Missing Values ---
    # Fill numeric columns using Genre-wise mean
    for col in ['Episode_Length_minutes', 'Guest_Popularity_percentage']:
        combined[col] = combined.groupby('Genre')[col].transform(lambda x: x.fillna(x.mean()))

    # --- Encode Categorical Features ---
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day',
                        'Publication_Time', 'Episode_Sentiment', 'length_bucket', 'genre_sentiment']

    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    # Split back to train and test
    X_train_fe = combined.iloc[:len(X_train)].reset_index(drop=True)
    X_test_fe = combined.iloc[len(X_train):].reset_index(drop=True)

    return X_train_fe, X_test_fe


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
target = 'Listening_Time_minutes'


train.head()


train.info()


train['Episode_Length_minutes'].hist()


train['Guest_Popularity_percentage'].hist()


train2, test2= engineer_features(train, test)


train2


# train.select_dtypes(include=object).columns.to_list()


# #categorical encording
# list_cols = train.select_dtypes(include=object).columns.to_list() #対象列指定
# cate_Encoder =  ce.OrdinalEncoder(cols=list_cols, drop_invariant=True)
# cate_Ordinal =  cate_Encoder.fit_transform(train[list_cols]) #実行
# cate_Ordinal_test = cate_Encoder.transform(test[list_cols])

# train2 = pd.concat([cate_Ordinal,train.select_dtypes(exclude=object)], axis=1)
# test2 = pd.concat([cate_Ordinal_test,test.select_dtypes(exclude=object)], axis=1)
# train2


test2


train2 = train2.fillna(train2.median())
train2


train2.shape


test2.shape


train2['Listening_Time_minutes'].hist(alpha=0.5)
train2['Guest_Popularity_percentage'].hist(alpha=0.5)


plt.figure(figsize=(20, 10))
sns.heatmap(train2.corr(),annot=True)



#modeling
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import itertools

def lightgbm_tuning(train_x, train_y):
    # パラメータの探索空間
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'boosting_type': ['gbdt'],
        'objective': ['regression'],
        'metric': ['rmse'],
        'max_depth': [3, 4, 5]
    }
    
    # 全ての組み合わせのリストを作成
    all_params = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]
    
    best_score = float('inf')
    best_params = None
    
    # 各パラメータセットでのモデルの訓練
    for params in all_params:
        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)
        
        # モデルの訓練
        train_data = lgb.Dataset(X_train, y_train)
        model = lgb.train(params, train_data, num_boost_round=100)
        
        # RMSEの計算
        test_preds = model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        
        # ベストスコアの更新
        if test_rmse < best_score:
            best_score = test_rmse
            best_params = params
            
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {best_score:.4f}")
    
    return best_params
    
train_y = train2[target]
train_X = train2.drop(target,axis=1)
test2 = test2.drop(target,axis=1)
best_alpha = lightgbm_tuning(train_X, train_y)


best_alpha


train_data = lgb.Dataset(train_X, train_y)
model = lgb.train(best_alpha, train_data, num_boost_round=200)



pred = model.predict(test2)


submission.shape


submission.head()


pred


submission['Listening_Time_minutes'] = pred
sub = submission[['id','Listening_Time_minutes']]

sub.to_csv('submission.csv', index=False)
sub.head(3)




