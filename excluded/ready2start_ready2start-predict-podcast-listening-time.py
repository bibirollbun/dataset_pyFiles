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


from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import OrdinalEncoder


# from sklearn.metrics import root_mean_squared_error
import sklearn
print('ScikitLearn Version :', sklearn.__version__)

# root_mean_squared_error는 ScikitLearn 1.4 버전에 추가되었지만, 현재 버전은 1.2.2
# mean_squared_error(squared=False)로 RMSE 계산

from sklearn.metrics import mean_squared_error


train_data = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# V1 ~ V4. Lasso 모델 적용 (하이퍼파라미터 튜닝)
# V5. XGBoost 모델 적용
# V6 ~ V8. RandomForest 모델 적용
#  - V8 : Number_of_Ads 이상치 처리
# V9. Voting (RandomForest, XGBoost, GradientBoostingRegressor)


train_data.drop(['id'], axis=1, inplace=True)
test_data.drop(['id'], axis=1, inplace=True)


# 결측치 제거
train_data = train_data[train_data['Episode_Length_minutes'].notnull()]
train_data = train_data[train_data['Number_of_Ads'].notnull()]


train_data.head(3)


train_data[['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads', 'Listening_Time_minutes']].corr()

# corr
# Episode_Length_minute와 매우 높은 양의 상관관계
# Number_of_Ads와 음의 상관관계
# Host_Popularity_percentage와 양의 상관관계


# Guest_Popularity_percentage 결측치 평균값으로 대체
mean_guest_popularity_percentage = train_data['Guest_Popularity_percentage'].mean()
train_data.fillna(mean_guest_popularity_percentage, inplace=True)


y = train_data['Listening_Time_minutes']
x = train_data.iloc[:, :-1]
x.head(3)


x['Number_of_Ads'].value_counts()


# V8 : Number_of_Ads 이상치 처리 => 1로 일괄 변경
x.loc[x['Number_of_Ads'] > 10, 'Number_of_Ads'] = 1
x['Number_of_Ads'].value_counts()


# 트리 모델은 범주 간 순서를 의미로 해석하지 않는다. 따라서 one-hot encoding 대신 label encoding 적용
x_le = x.copy()

df_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
enc = OrdinalEncoder()
x_le[df_cols] = enc.fit_transform(x_le[df_cols])
x_le


x_le.describe()


# Episode_Length_minutes와 다른 요소의 상관관계 => Number_of_Ads, Episode_Sentiment, Host_Popularity_percentage
print(x_le.corr()['Episode_Length_minutes'])


train_x, valid_x, train_y, valid_y = train_test_split(x_le, y, test_size=0.2, random_state=0)


# V9 : Voting (RandomForest, XGBoost, GradientBoostingRegressor)
rfRegressor = RandomForestRegressor(
    random_state = 0, 
    n_jobs = -1,
    verbose = 2,
    max_depth = None,
    n_estimators = 500,
)

xgbRegressor = xgb.XGBRegressor(
    n_estimators = 100,
)

gbRegressor = GradientBoostingRegressor(
    n_estimators = 100,
)

votingRegressor = VotingRegressor(estimators=[
    ('rfRegressor', rfRegressor), 
    ('xgbRegressor', xgbRegressor), 
    ('gbRegressor', gbRegressor)
])

votingRegressor.fit(train_x, train_y)


# RMSE 계산
# train_y_pred = votingRegressor.predict(train_x)
# valid_y_pred = votingRegressor.predict(valid_x)

# print('train data RMSE:', mean_squared_error(train_y, train_y_pred, squared=False))
# print('validation data RMSE:', mean_squared_error(valid_y, valid_y_pred, squared=False))


new_test_data = test_data.copy()
new_test_data.head(3)


# new_test_data.describe()


# Episode_Length_minutes 이상치 처리
new_test_data.loc[54434, 'Episode_Length_minutes'] = train_data.loc[train_data['Podcast_Name'] == 'Current Affairs', 'Episode_Length_minutes'].mean()
new_test_data.loc[56597, 'Episode_Length_minutes'] = train_data.loc[train_data['Podcast_Name'] == 'Market Masters', 'Episode_Length_minutes'].mean()

new_test_data[new_test_data['Episode_Length_minutes'] > 200]


new_test_data['Number_of_Ads'].value_counts()


# V8 : Number_of_Ads 이상치 처리 => 1로 일괄 변경
new_test_data.loc[new_test_data['Number_of_Ads'] > 10, 'Number_of_Ads'] = 1
new_test_data['Number_of_Ads'].value_counts()


# Episode_Length_minutes => 팟캐스트, 에피소드별 평균값 채우기 
new_test_data['Episode_Length_minutes'] = new_test_data['Episode_Length_minutes'].fillna(
    new_test_data.groupby(['Podcast_Name', 'Episode_Title'])['Episode_Length_minutes'].transform('mean')
)

# V7 : 팟캐스트별 평균값으로 수정 (V8에서 원복)
# new_test_data['Episode_Length_minutes'] = test_data['Episode_Length_minutes'].fillna(
#     test_data.groupby(['Podcast_Name'])['Episode_Length_minutes'].transform('mean')
# )

# Guest_Popularity_percentage => 전체 평균값 (train_data 기준) 채우기
new_test_data['Guest_Popularity_percentage'] = mean_guest_popularity_percentage
new_test_data.isnull().sum()


# 테스트 데이터 Label Encoding
x_test_le = new_test_data.copy()

x_test_le[df_cols] = enc.transform(x_test_le[df_cols])
x_test_le.head(3)


prediction = votingRegressor.predict(x_test_le)
submission['Listening_Time_minutes'] = prediction


submission.to_csv('submission.csv', index=False)




