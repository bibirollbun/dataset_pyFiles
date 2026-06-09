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


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
import scipy


train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")


train.columns #데이터 피처 이해


test.columns #데이터 피처 이해


train.shape #데이터 양 확인


test.shape #데이터 양 확인


submission.shape #데이터 양 확인


train.head() #데이터 대략적으로 확인


test.head() #데이터 대략적으로 확인


submission.head() #데이터 대략적으로 확인


train.dtypes #데이터 형태 확인


test.dtypes #데이터 형태 확인


# datetime 컬럼을 datetime 자료형으로 변환
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# 연도, 월, 일, 시간(시간대), 요일 추출
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day              # ✅ 추가된 부분
train['hour'] = train['datetime'].dt.hour
train['dayofweek'] = train['datetime'].dt.dayofweek  # 월:0 ~ 일:6

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day                # ✅ 추가된 부분
test['hour'] = test['datetime'].dt.hour
test['dayofweek'] = test['datetime'].dt.dayofweek


train[['datetime', 'year', 'day', 'month', 'hour', 'dayofweek']].head() #데이터 변환 점검


train.columns


train.describe()


sns.barplot(data = train, x = 'year', y = 'count') #daytime-year 변수 시각화


sns.barplot(data = train, x = 'month', y = 'count') #daytime-month 시각화


 sns.barplot(data = train, x = 'season', y = 'count')


for s in range(1, 5):
    months = sorted(train[train['season'] == s]['month'].unique())
    print(f"Season {s}: {months}")

#season month 매칭 확인


# 하나의 subplot을 만들고 크기 지정
fig, ax1 = plt.subplots(1, 1, figsize=(20, 5))

# 포인트 플롯 그리기
sns.pointplot(data=train, x='hour', y='count', ax=ax1)

#hour변수 시각화


# 1행 1열짜리 subplot 만들기 (단일 그래프)
fig, ax1 = plt.subplots(1, 1, figsize=(20, 5)) 

# Seaborn 포인트플롯: workingday(출근일 여부)로 나눠서 시각화
sns.pointplot(data=train, x='hour', y='count', hue='workingday', ax=ax1)

#hour과 workingday 시각화 


fig, ax1 = plt.subplots(1, 1, figsize=(20, 5)) 

sns.pointplot(data=train, x='hour', y='count', hue='holiday', ax=ax1)

#hour과 holiday 시각화 


fig, ax1 = plt.subplots(1, 1, figsize=(20, 5)) 

sns.pointplot(data=train, x='hour', y='count', hue='weather', ax=ax1)

#시간대에 따른 날씨 체크
#weather4가 점처럼 찍히는 것으로 보아, 이상값인듯?


train['weather'].value_counts() #weather4 이상값인지 확인 -> 제거 필요


# 1. weather == 4 제거
train_filtered = train[train['weather'] != 4]

# 2. 필터링된 데이터로 그래프 그리기
fig, ax1 = plt.subplots(1, 1, figsize=(20, 5)) 
sns.pointplot(data=train_filtered, x='hour', y='count', hue='weather', ax=ax1)


fig, ax1 = plt.subplots(1, 1, figsize=(20, 5)) 

sns.pointplot(data=train, x='hour', y='count', hue='dayofweek', ax=ax1)
#요일 시각화


fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize=(12,5))

sns.scatterplot(data = train, x = 'windspeed', y = 'count', ax = ax1)
sns.scatterplot(data = train, x = 'temp', y = 'count', ax = ax2)
sns.scatterplot(data = train, x = 'humidity', y =  'count', ax = ax3)


len(train[train['windspeed']==0]) #windspeed가 0인 경우가 많음 -> 측정되지 않은 경우에 모두 0으로 처리된 듯 


correlation_matrix = train.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.show()


fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows = 6, figsize = (12,10))
sns.boxplot(data = train, x = 'windspeed', ax = ax1)
sns.boxplot(data = train, x = 'humidity', ax = ax2)
sns.boxplot(data = train, x = 'temp', ax = ax3)
sns.boxplot(data = train, x = 'casual', ax = ax4)
sns.boxplot(data = train, x = 'registered', ax = ax5)
sns.boxplot(data = train, x = 'count', ax = ax6)

#연속형 변수 이상치 제거


train.shape


from collections import Counter
import numpy as np

def detect_outliers(data, n, cols):
    outlier_indices = []
    for col in cols:
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        
        outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    
    return multiple_outliers

Outliers_to_drop = detect_outliers(train, 2, ["temp", "atemp", "casual", "registered", "humidity", "windspeed", "count"])

#IQR방식을 사용한 이상치 제거



train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop = True)
train.shape


train['count_log'] = train['count'].map(lambda x: np.log(x) if x > 0 else 0)

# 2) 무한대값 처리
train['count_log'] = train['count_log'].replace([np.inf, -np.inf], np.nan)

# 3) NaN 제거
train = train.dropna(subset=['count_log'])

# 4) 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.histplot(train['count_log'], kde=True, color='b')

plt.title('Distribution of Log-transformed Count')
plt.xlabel('Log(count)')
plt.ylabel('Frequency')

# 5) 왜도와 첨도 출력
print("skewness(왜도): %f" % train['count_log'].skew())
print("kurtosis(첨도): %f" % train['count_log'].kurt())

plt.show()

#count에 log취해서 정규화 시켜주기
#원본 count를 어떻게 하다보니 실수로 삭제함...


from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def predict_windspeed(data):
    data = data.copy()
    
    wind0 = data[data['windspeed'] == 0].copy()
    windnot0 = data[data['windspeed'] != 0].copy()
    
    # 예측할 wind0이 비어 있으면 원본 그대로 반환
    if wind0.empty:
        return data

    col = ['season', 'weather', 'temp', 'humidity', 'atemp', 'day']

    # 문자열로 변환해서 분류 문제로 처리
    windnot0['windspeed'] = windnot0['windspeed'].astype(str)

    rf = RandomForestClassifier()
    rf.fit(windnot0[col], windnot0['windspeed'])

    pred_wind0 = rf.predict(wind0[col])
    wind0['windspeed'] = pred_wind0

    # float으로 되돌리기
    windnot0['windspeed'] = windnot0['windspeed'].astype(float)
    wind0['windspeed'] = wind0['windspeed'].astype(float)

    data = pd.concat([windnot0, wind0], axis=0)
    data.reset_index(drop=True, inplace=True)

    return data
#windspeed = 0 대체값 찾기


train = predict_windspeed(train)
test = predict_windspeed(test)


train[train['windspeed'] == 0.0]


fig, (ax1, ax2) = plt.subplots(2,1)
fig.set_size_inches(20,15)

#갯수를 세야하니 countplot
sns.countplot(data = train, x = 'windspeed', ax = ax1)
sns.countplot(data = test, x = 'windspeed', ax = ax2)


categorical_feature_names = ["season",'holiday','weather',
                             'dayofweek','month','year','hour']


for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category") 


train.dtypes


test.dtypes


feature_names =['season','holiday','weather','temp','humidity','windspeed','year','month','hour','dayofweek']


X_train = train[feature_names]

print(X_train.shape)
X_train.head()


X_test = test[feature_names]

print(X_test.shape)
X_test.head()


label_name = 'count_log'

y_train = train[label_name]

print(y_train.shape)
y_train.head()


from sklearn.metrics import make_scorer

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


from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

train_input = train[feature_names].values
train_target = train['count_log'].values

kfold = KFold(n_splits=10, random_state=42, shuffle=True)
model = GradientBoostingRegressor(random_state=42)

scores = cross_validate(model, train_input, train_target, cv=kfold,
                        return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))


# 교차검증으로 평가
scores = cross_validate(model, train_input, train_target, cv=kfold,
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))

# 실제 예측을 위한 학습
model.fit(train_input, train_target)

# 테스트셋 예측
test_input = test[feature_names].values
pred = model.predict(test_input)

submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

# 예측 결과를 count_log 컬럼에 넣기
submission['count_log'] = pred

# 로그 변환된 값을 다시 원래 스케일로 변환 (지수 변환)
submission['count'] = np.exp(submission['count_log'])

# count_log 컬럼은 제거
submission.drop('count_log', axis=1, inplace=True)

# 제출용 CSV 파일 저장 (Kaggle 노트북 작업 디렉토리에 저장)
submission.to_csv("bike_submission.csv", index=False)


submission.head()

