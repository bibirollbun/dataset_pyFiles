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


!pip install pytorch-tabnet


import csv
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetRegressor
import torch


def engineer_features(X_train, X_test):
    combined = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

    # outlier removal
    combined['Episode_Length_minutes'] = np.maximum(0, np.minimum(120, combined['Episode_Length_minutes']))
    combined['Host_Popularity_percentage'] = np.maximum(20, np.minimum(100, combined['Host_Popularity_percentage']))
    combined['Guest_Popularity_percentage'] = np.maximum(0, np.minimum(100, combined['Guest_Popularity_percentage']))
    combined.loc[combined['Number_of_Ads'] > 3, 'Number_of_Ads'] = 0

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
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
target = 'Listening_Time_minutes'


train2, test2= engineer_features(train, test)


train2 = train2.fillna(train2.median())
test2 = test2.fillna(test2.median())


train_id = train2['id']
test_id = test['id']
train2.drop('id',axis = 1)
test2.drop('id',axis =1)


# scaler = StandardScaler()
# train3 = scaler.fit_transform(train2)
# test3 = scaler.transform(test2)
test2


train2['Guest_Popularity_percentage'].plot(kind='hist')


plt.hist(train2['episode_number'],alpha=0.5)
plt.hist(test2['episode_number'],alpha=0.5)
plt.show()


train2.groupby('episode_number').count()


train2.describe()


test2.describe()


train3, valid3 = train_test_split(train2, test_size = 0.2, random_state = 0)


#学習用データ
y_train = train3[target].values.reshape(-1, 1)
X_train = train3.drop(target,axis=1)
X_test = test2.drop(target,axis=1)

#検証用データ
y_val = valid3[target].values.reshape(-1, 1)
X_val = valid3.drop(target,axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.fit_transform(X_test)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape)


PARAMS = dict(
    seed = 0, # random seed num
    optimizer_params = dict(lr = 1e-2), # set learning rate
    verbose = 10, # learning report per 10
)

# modeling
model = TabNetRegressor(**PARAMS)

# learning and eval
model.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)],#evaluation dataset
    eval_metric= ['rmse'],
    batch_size = 32,  # 
    max_epochs = 100, # the number of max learning times
    patience = 10,    # early_stoppping(0 is nothing)
)


import matplotlib.pyplot as plt
plt.plot(model.history["loss"], label = "loss")
plt.show()


pred = model.predict(X_test)
pred


submission[target]=pred


submission.head()


plt.plot(model.history["val_0_rmse"], label = "test")
plt.show()


submission.to_csv('submission.csv', index=False)

