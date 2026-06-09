import pandas as pd
# 데이터 경로
data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')


(train.shape, test.shape)


train.head()


test.head()


submission.head()


# 훈련 데이터에서 weather가 4가 아닌 데이터만 추출
train = train[train['weather'] != 4]


train.shape


all_data_temp = pd.concat([train, test])
all_data_temp


all_data = pd.concat([train, test], ignore_index=True)
all_data


# datetime 타입으로 바꾸기
all_data['datetime'] = pd.to_datetime(all_data['datetime']) 

all_data['year'] = all_data['datetime'].dt.year # 연도
all_data['month'] = all_data['datetime'].dt.month # 월
all_data['hour'] = all_data['datetime'].dt.hour # 시간
all_data["weekday"] = all_data['datetime'].dt.weekday # 요일


all_data.head()


drop_features = ['casual', 'registered', 'datetime', 'month', 'windspeed']

all_data = all_data.drop(drop_features, axis=1)


# 훈련 데이터와 테스트 데이터 나누기
X_train = all_data[~pd.isnull(all_data['count'])]
X_test = all_data[pd.isnull(all_data['count'])]

# 타깃값 count 제거
X_train = X_train.drop(['count'], axis=1)
X_test = X_test.drop(['count'], axis=1)

y = train['count'] # 타깃값


X_train.head()


import numpy as np

def rmsle(y_true, y_pred, convertExp=True):
    # 지수변환 -> y를 로그변환 했으므로 (분석 정리 11) 다시 이를 지수변환 해주어야 타깃값인 'count' 복원 가능
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)
        
    # 로그변환 후 결측값을 0으로 변환
    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))
    
    # RMSLE 계산
    output = np.sqrt(np.mean((log_true - log_pred)**2))
    return output


from sklearn.linear_model import LinearRegression

linear_reg_model = LinearRegression()


log_y = np.log(y)  # 타깃값 로그변환
linear_reg_model.fit(X_train, log_y) # 모델 훈련


preds = linear_reg_model.predict(X_train)


print (f'선형회귀의 RMSLE 값 : {rmsle(log_y, preds, True):.4f}')


linearreg_preds = linear_reg_model.predict(X_test) # 테스트 데이터로 예측

submission['count'] = np.exp(linearreg_preds)    # 지수변환
submission.to_csv('submission_base.csv', index=False) # 파일로 저장


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


## RMSLE 계산 함수
def rmsle(y_true, y_pred, convertExp=True):
    # 지수변환 -> y를 로그변환 했으므로 (분석 정리 11) 다시 이를 지수변환 해주어야 타깃값인 'count' 복원 가능
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)
        
    # 로그변환 후 결측값을 0으로 변환
    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))
    
    # RMSLE 계산
    output = np.sqrt(np.mean((log_true - log_pred)**2))
    return output

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)  # RMSLE를 최적화 점수로 사용


## 랜덤 포레스트 모델 및 그리드 서치 적용
rf = RandomForestRegressor(random_state=42)

# 그리드 서치의 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [50, 100, 200],  # 트리 개수
    'max_depth': [10, 20],       # 최대 깊이
    'min_samples_split': [5, 10], # 분할 최소 샘플 수
    'min_samples_leaf': [2, 4]    # 리프 노드의 최소 샘플 수
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=rmsle_scorer,  # RMSLE를 사용하여 최적화
    cv=3,  # 3-폴드 교차 검증
    n_jobs=-1,  # 병렬 실행
    verbose=0  # 검증 로그 결과 출력하지 않음
)

grid_search.fit(X_train, log_y)

# 최적의 하이퍼파라미터 출력
print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")


## 최적 모델 평가 및 예측
best_rf = grid_search.best_estimator_

# 훈련 데이터에서 RMSLE 평가
train_preds = best_rf.predict(X_train)
print (f'랜덤 포레스트 회귀의 RMSLE 값 : {rmsle(log_y, train_preds, True):.4f}')


import seaborn as sns
import matplotlib.pyplot as plt

randomforest_preds = best_rf.predict(X_test)

figure, axes = plt.subplots(ncols=2)
figure.set_size_inches(10, 4)

sns.histplot(y, bins=50, ax=axes[0])
axes[0].set_title('Train Data Distribution')
sns.histplot(np.exp(randomforest_preds), bins=50, ax=axes[1])
axes[1].set_title('Predicted Test Data Distribution');


# 테스트 데이터 예측
test_preds = np.exp(best_rf.predict(X_test))  # 로그 변환된 값 원래 값으로 복구

# 제출 파일 생성
submission['count'] = test_preds
submission.to_csv('submission_rf_gs.csv', index=False)

print("최적의 랜덤 포레스트 모델을 사용한 예측 완료 및 제출 파일 생성됨.")


train = pd.read_csv(data_path + 'train.csv', parse_dates=["datetime"])
test = pd.read_csv(data_path + 'test.csv', parse_dates=["datetime"])

all_data = pd.concat([train, test], ignore_index=True)

all_data['year'] = all_data['datetime'].dt.year # 연도
all_data['month'] = all_data['datetime'].dt.month # 월
all_data['hour'] = all_data['datetime'].dt.hour # 시간
all_data["weekday"] = all_data['datetime'].dt.weekday # 요일


# widspeed 풍속에 0 값이 가장 많다. => 잘못 기록된 데이터를 고쳐 줄 필요가 있음
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18,10)

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count',title="train windspeed")
sns.countplot(data=train, x="windspeed", ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(ylabel='Count',title="test windspeed")
sns.countplot(data=test, x="windspeed", ax=axes[1])


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. 데이터 로드
data_path = '/kaggle/input/bike-sharing-demand/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')

# 2. 데이터 합치기 (train/test를 구분하기 위해 'set' 컬럼 추가)
train['set'] = 'train'
test['set'] = 'test'
all_data = pd.concat([train, test], ignore_index=True)

# 3. 날짜 변환 및 피처 생성
all_data['datetime'] = pd.to_datetime(all_data['datetime'])
all_data['year'] = all_data['datetime'].dt.year
all_data['month'] = all_data['datetime'].dt.month
all_data['hour'] = all_data['datetime'].dt.hour
all_data['weekday'] = all_data['datetime'].dt.weekday

# 4. windspeed 예측을 위한 데이터셋 생성 (0값을 예측하여 대체)
features = ['humidity', 'season', 'month', 'hour']
windspeed_data = all_data[features + ['windspeed']]

# windspeed가 0이 아닌 데이터 → 훈련 데이터로 사용
windspeed_train = windspeed_data[windspeed_data['windspeed'] != 0]
X_train_wind = windspeed_train[features]
y_train_wind = windspeed_train['windspeed']

# windspeed가 0인 데이터 → 예측 대상
windspeed_test = windspeed_data[windspeed_data['windspeed'] == 0]
X_test_wind = windspeed_test[features]

# 5. 랜덤 포레스트 모델 학습
wind_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
wind_model.fit(X_train_wind, y_train_wind)

# 6. windspeed 0인 값 예측 및 대체
predicted_windspeed = wind_model.predict(X_test_wind)
all_data.loc[all_data['windspeed'] == 0, 'windspeed'] = predicted_windspeed

# 7. windspeed 보정 완료 확인
print("windspeed 0값을 머신러닝 모델을 이용해 보완 완료!")
print(all_data['windspeed'].describe())  # 통계 요약 확인


# widspeed 의 이상치(0값)을 조정한 데이터를 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# windspeed 분포 확인 (수정 코드)
fig, ax1 = plt.subplots()
fig.set_size_inches(18,6)

plt.sca(ax1)
ax1.set(xlabel='windspeed', ylabel='count', title="Train windspeed distribution")

sns.histplot(data=all_data, x="windspeed", bins=30, kde=True, ax=ax1)  # 연속형 변수 시각화
plt.show()


# 풍속이 0인것과 아닌 것의 세트를 나누어 준다.
trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed'] != 0]
print(trainWind0.shape)
print(trainWindNot0.shape)


# 그래서 머신러닝으로 예측을 해서 풍속을 넣어주도록 한다.
from sklearn.ensemble import RandomForestClassifier

def predict_windspeed(data):
    
    # 풍속이 0인것과 아닌 것을 나누어 준다.
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWindNot0 = data.loc[data['windspeed'] != 0]
    
    # 풍속을 예측할 피처를 선택한다.
    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

    # 풍속이 0이 아닌 데이터들의 타입을 스트링으로 바꿔준다.
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

    # 랜덤포레스트 분류기를 사용한다.
    rfModel_wind = RandomForestClassifier()

    # wCol에 있는 피처의 값을 바탕으로 풍속을 학습시킨다.
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])

    # 학습한 값을 바탕으로 풍속이 0으로 기록 된 데이터의 풍속을 예측한다.
    wind0Values = rfModel_wind.predict(X = dataWind0[wCol])

    # 값을 다 예측 후 비교해 보기 위해
    # 예측한 값을 넣어 줄 데이터 프레임을 새로 만든다.
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0

    # 값이 0으로 기록 된 풍속에 대해 예측한 값을 넣어준다.
    predictWind0["windspeed"] = wind0Values

    # dataWindNot0 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합쳐준다.
    data = predictWindNot0.append(predictWind0)

    # 풍속의 데이터타입을 float으로 지정해 준다.
    data["windspeed"] = data["windspeed"].astype("float")

    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    
    return data


# 1. 데이터 로드
data_path = '/kaggle/input/bike-sharing-demand/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')

# 훈련 데이터에서 weather가 4가 아닌 데이터만 추출
train = train[train['weather'] != 4]

# 2. 데이터 합치기 (train/test를 구분하기 위해 'set' 컬럼 추가)
train['set'] = 'train'
test['set'] = 'test'
all_data = pd.concat([train, test], ignore_index=True)

# 3. 날짜 변환 및 피처 생성
all_data['datetime'] = pd.to_datetime(all_data['datetime'])
all_data['year'] = all_data['datetime'].dt.year
all_data['month'] = all_data['datetime'].dt.month
all_data['hour'] = all_data['datetime'].dt.hour
all_data['weekday'] = all_data['datetime'].dt.weekday

# 0값을 조정한다.
all_data = predict_windspeed(all_data)

# widspeed 의 0값을 조정한 데이터를 시각화
fig, ax1 = plt.subplots()
fig.set_size_inches(18,6)

plt.sca(ax1)
plt.xticks(rotation=30, ha='right')
ax1.set(ylabel='Count',title="train windspeed")
sns.countplot(data=all_data, x="windspeed", ax=ax1)


drop_features = ['casual', 'registered', 'datetime', 'month', 'set']

all_data = all_data.drop(drop_features, axis=1)


# 훈련 데이터와 테스트 데이터 나누기
X_train = all_data[~pd.isnull(all_data['count'])]
X_test = all_data[pd.isnull(all_data['count'])]

# 타깃값 count 제거
X_train = X_train.drop(['count'], axis=1)
X_test = X_test.drop(['count'], axis=1)

y = train['count'] # 타깃값
log_y = np.log(y)  # 타깃값 로그변환


X_train.shape, X_test.shape, y.shape


X_train


## 랜덤 포레스트 모델 및 그리드 서치 적용
rf_ip = RandomForestRegressor(random_state=42)

# 그리드 서치의 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [50, 100, 200],  # 트리 개수
    'max_depth': [10, 20],       # 최대 깊이
    'min_samples_split': [5, 10], # 분할 최소 샘플 수
    'min_samples_leaf': [2, 4]    # 리프 노드의 최소 샘플 수
}

grid_search_ip = GridSearchCV(
    estimator=rf_ip,
    param_grid=param_grid,
    scoring=rmsle_scorer,  # RMSLE를 사용하여 최적화
    cv=3,  # 3-폴드 교차 검증
    n_jobs=-1,  # 병렬 실행
    verbose=0  # 검증 로그 결과 출력하지 않음
)

grid_search_ip.fit(X_train, log_y)

# 최적의 하이퍼파라미터 출력
print(f"최적의 하이퍼파라미터: {grid_search_ip.best_params_}")


# 예측
preds = grid_search_ip.best_estimator_.predict(X_train)

# 평가
print(f'랜덤 포레스트 회귀+보간법 RMSLE 값 : {rmsle(log_y, preds, True):.4f}')


submission['count'] = np.exp(randomforest_preds) # 지수변환
submission.to_csv('submission_rf_ip.csv', index=False)

