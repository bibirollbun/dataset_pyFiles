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


# 라이브러리 호출
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import make_scorer # RMSLE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

import warnings
warnings.filterwarnings(action='ignore') 


df_train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv') # 훈련 데이터
df_test = pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')   # 테스트 데이터
df_submission = pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv') # 제출 샘플 데이터


# 컬럼 속성 확인기기

print('=' * 20, 'Train Data Info', '=' * 20)
print(df_train.shape)
print(df_train.head())

print('')

print('=' * 20, 'Test Data Info', '=' * 20)
print(df_test.shape)
print(df_test.head())

print('')

print('=' * 20, 'Gender Data Info', '=' * 20)
print(df_submission.shape)
print(df_submission.head())


df_test.head()


df_submission.head()


df_train.info()

# datetime 형태로 변경
df_train['datetime'] = pd.to_datetime(df_train['datetime'])
df_test['datetime'] = pd.to_datetime(df_test['datetime'])
df_train.info()


df_train.isnull().sum()


df_train.describe()


# datetime'을 연, 월, 일, 시, 분, 초 로 나누어 컬럼에 추가

df_train['year'] = df_train['datetime'].dt.year
df_train['month'] = df_train['datetime'].dt.month
df_train['day'] = df_train['datetime'].dt.day
df_train['hour'] = df_train['datetime'].dt.hour
df_train['minute'] = df_train['datetime'].dt.minute
df_train['second'] = df_train['datetime'].dt.second
#월(0) 화(1) 수(2) 목(3) 금(4) 토(5) 일(6)
df_train['dayofweek'] = df_train['datetime'].dt.dayofweek

df_test['year'] = df_test['datetime'].dt.year
df_test['month'] = df_test['datetime'].dt.month
df_test['day'] = df_test['datetime'].dt.day
df_test['hour'] = df_test['datetime'].dt.hour
df_test['minute'] = df_test['datetime'].dt.minute
df_test['second'] = df_test['datetime'].dt.second
df_test['dayofweek'] = df_test['datetime'].dt.dayofweek

df_train.head()


# 연도별
sns.barplot(data = df_train, x='year', y='count', palette="Set2")


# 연도-월 별로 더 세분화 해보기 위해 새로운 컬럼 year_month를 만들기

def con_year_month(datetime):
    return "{0}-{1}".format(datetime.year, datetime.month)

df_train["year_month"] = df_train["datetime"].apply(con_year_month)
df_test["year_month"] = df_test["datetime"].apply(con_year_month)

print(df_train.shape)
df_train[["datetime", "year_month"]].head()


plt.figure(figsize=(15, 5))
sns.barplot(x='year_month', y='count', data=df_train)


sns.barplot(data = df_train, x='month', y='count')


# 요일별
sns.barplot(data = df_train, x='dayofweek', y='count')


# 계절별
sns.barplot(data = df_train, x='season', y='count', palette="Set2")


# season이 월별로 어떻게 나뉘었는지 확인
pd.crosstab(df_train['season'], df_train['month'])


# 시간대별
plt.figure(figsize=(15, 5))
sns.pointplot(x='hour', y='count', data=df_train)


# 시간대별로 근무 여부/요일/날씨/계절에 따라 어떤 차이가 있는지 확인

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

sns.pointplot(x='hour', y='count', hue='workingday', data=df_train, ax=axes[0], palette="Set2")
sns.pointplot(x='hour', y='count', hue='dayofweek', data=df_train, ax=axes[1], palette="Set2")


# 상관계수
cor_train = df_train[["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]]
cor_train.corr()


plt.figure(figsize=(15, 10))
sns.heatmap(cor_train.corr(), annot=True, cmap='Blues')


# 온도, 바람세기, 습도 데이터 확인

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))

sns.regplot(data = df_train, x='temp', y='count', ax = ax1)
sns.regplot(data = df_train, x='windspeed', y='count', ax = ax2)
sns.regplot(data = df_train, x='humidity', y='count', ax = ax3)


# windspeed 컬럼만 확인

plt.figure(figsize=(15, 5))
sns.countplot(data = df_train, x='windspeed')


# 이상치 제거

fig, axes = plt.subplots(6, 1, figsize = (12, 8))

sns.boxplot(data = df_train, x="temp", ax=axes[0])
sns.boxplot(data = df_train, x="humidity", ax=axes[1])
sns.boxplot(data = df_train, x="windspeed", ax=axes[2])
sns.boxplot(data = df_train, x="casual", ax=axes[3])
sns.boxplot(data = df_train, x="registered", ax=axes[4])
sns.boxplot(data = df_train, x="count", ax=axes[5])


# 이상치 제거 후 분포 확인

cols = ['humidity', 'windspeed', 'casual', 'registered', 'count']
for col in cols:
    df_train = df_train[np.abs(df_train[col] - df_train[col].mean()) <= (3*df_train[col].std())]
    
fig, axes = plt.subplots(6, 1, figsize = (12, 8))

sns.boxplot(data = df_train, x="temp", ax=axes[0])
sns.boxplot(data = df_train, x="humidity", ax=axes[1])
sns.boxplot(data = df_train, x="windspeed", ax=axes[2])
sns.boxplot(data = df_train, x="casual", ax=axes[3])
sns.boxplot(data = df_train, x="registered", ax=axes[4])
sns.boxplot(data = df_train, x="count", ax=axes[5])

print(df_train.shape)  #(10886, 12) 변경 전
print(df_train.shape)  #(10212, 20) 변경 후


# 정규화

# count 값의 데이터 분포 파악
figure, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.distplot(df_train['count'], ax=axes[0][0])
stats.probplot(df_train['count'], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(df_train['count']), ax=axes[1][0])
stats.probplot(np.log1p(df_train['count']), dist='norm', fit=True, plot=axes[1][1])


df_train['count_log'] = df_train['count'].map(lambda i:np.log(i) if i > 0 else 0)
df_train.head()


# 풍속이 0인 데이터 처리

figure, axes = plt.subplots(2, 1, figsize=(12, 8))

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(title='train windspeed')
sns.countplot(data = df_train, x='windspeed', ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(title='test windspeed')
sns.countplot(data = df_test, x='windspeed', ax=axes[1])


# windspeed가 0인 값을 랜덤포레스트를 이용해 예측한 값으로 대체

def predict_windspeed(data):
    wind0    = data[data["windspeed"] == 0].copy()
    windnot0 = data[data["windspeed"] != 0].copy()

    # wind0에 대한 예측 로직 (생략)
    # wind0["windspeed"] = model.predict(wind0[feature_cols])

    # wind0와 windnot0를 합치기
    data = pd.concat([windnot0, wind0], axis=0)

    # 타입 정리 및 인덱스 리셋
    data["windspeed"] = data["windspeed"].astype(float)
    data = data.sort_index().reset_index(drop=True)
    return data

# 적용 예
df_train = predict_windspeed(df_train)
df_test  = predict_windspeed(df_test)


df_train[df_train['windspeed'] == 0]

figure, axes = plt.subplots(2, 1, figsize=(12, 8))

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(title='train windspeed')
sns.countplot(data = df_train, x='windspeed', ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(title='test windspeed')
sns.countplot(data = df_test, x='windspeed', ax=axes[1])


# 범주형 변수 변경 - Category 형태로

cols = ['season', 'holiday', 'workingday', 'weather', 'dayofweek', 'month', 'hour', 'year']

for col in cols:
    df_train[col] = df_train[col].astype('category')
    df_test[col] = df_test[col].astype('category')
    
df_train.info()


# 모델링
# 변수 선택
# 피쳐로 사용할 변수를 리스트로 만들어준 뒤 train_input, train_target, test_input을 만들기
# arget으로는 'count'가 아닌 위에서 만들었던 'count_log'를 사용

features = ['season', 'holiday', 'weather', 'temp', 'humidity',
           'windspeed', 'year', 'month', 'hour', 'dayofweek']
           
train_input = df_train[features]
print(train_input.shape)
train_input.head()


test_input = df_test[features]
print(test_input.shape)
test_input.head()


label_name = 'count_log'

train_target = df_train[label_name]
print(train_target.shape)
train_target.head()


# RMSLE 함수
def RMSLE(predicted_values, actual_values):
    
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    difference = log_predict - log_actual
    difference = np.square(difference)
    
    mean_difference = difference.mean()
    score = np.sqrt(mean_difference)
    
    return score
 
rmsle_score = make_scorer(RMSLE)


# KFold
kfold = KFold(n_splits=10, random_state=42, shuffle=True)

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=42)
scores = cross_validate(model, train_input, train_target,
                       return_train_score=True, n_jobs=-1)

# train, val data 점수
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9194017180370215 0.9018757705894085


# RMSLE 점수
score = cross_val_score(model, train_input, train_target, cv=kfold, scoring=rmsle_score)
score = score.mean()
score  #0.13077674772551076

model.fit(train_input, train_target)

