# 데이터 로드
import pandas as pd
import numpy as np

train_df = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test_df = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
submission = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")



train_df.head(5)


test_df.head(5)


train_df.info()


train_df.describe()


# 각 컬럼의 유니크값 확인
print("======== 유니크값 개수")
print(train_df.nunique())

print("======== 각 변수 유니크값 상세 보기")
for column in train_df.columns:
    print(f"\n{column} unique values:")
    print(train_df[column].value_counts().sort_index())


# 날짜 데이터 처리 datetime 파싱
for df in [train_df, test_df]:
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 시간 관련 특성 추출
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if x in [8,9,17,18] else 0)
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

print(train_df[['datetime','year','month','hour','dayofweek']].head(5))
print(test_df[['datetime','year','month','hour','dayofweek']].head(5))


# 새로 생성된 시간 특성의 유니크값 확인
print("======== 생성된 시간 특성 유니크값")
time_features = ['year', 'month', 'day', 'hour', 'dayofweek', 'is_weekend']
for feature in time_features:
    print(f"\n{feature} unique values:")
    print(train_df[feature].value_counts().sort_index())


# 범주형 변수 처리  (원-핫 인코딩)
# 'holiday', 'workingday' 이미 이진화처리 되어 있음.

# train_df = pd.get_dummies(train_df, columns=['season', 'weather'])
# test_df = pd.get_dummies(test_df, columns=['season', 'weather'])

# 'weather' 컬럼에 대한 원-핫 인코딩
weather_encoded = pd.get_dummies(train_df['weather'], prefix='weather')
test_weather_encoded = pd.get_dummies(test_df['weather'], prefix='weather')

# 원본 데이터프레임에 인코딩된 컬럼 추가
train_df = pd.concat([train_df, weather_encoded], axis=1)
test_df = pd.concat([test_df, test_weather_encoded], axis=1)

# 'season' 컬럼에 대한 원-핫 인코딩
season_encoded = pd.get_dummies(train_df['season'], prefix='season')
test_season_encoded = pd.get_dummies(test_df['season'], prefix='season')

# 원본 데이터프레임에 인코딩된 컬럼 추가
train_df = pd.concat([train_df, season_encoded], axis=1)
test_df = pd.concat([test_df, test_season_encoded], axis=1)


# 결측치 확인
print("======== Train 결측치 확인")
print(train_df.isnull().sum())
print("======== Test 결측치 확인")
print(test_df.isnull().sum())


# 수치형 변수 전처리
for df in [train_df, test_df]:
    df['windspeed_log'] = np.log1p(df['windspeed'])
    df['temp_diff'] = df['temp'] - df['atemp']
    df['temp_humidity'] = df['temp'] * df['humidity']


!apt-get update -qq
!apt-get install fonts-nanum* -qq


!fc-cache -fv


# 수치형 변수 분포 확인
import matplotlib.pyplot as plt
import seaborn as sns


fig, axes = plt.subplots(2, 1, figsize=(12, 10))
sns.boxplot(data=train_df[['temp','atemp','humidity','windspeed']], ax=axes[0])
axes[0].set_title('Train Data Numerical Variable Distribution') # 한글 대신 영어로 제목 변경ㅜㅜ
sns.boxplot(data=test_df[['temp','atemp','humidity','windspeed']], ax=axes[1])
axes[1].set_title('Test Data Numerical Variable Distribution')
plt.tight_layout()
plt.show()


# train과 test의 컬럼 일치 확인
print("======== 컬럼 비교")
print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())


# 새로 생성된 특성들의 기술통계량 비교
print("======== 새로운 특성 통계량 비교")
new_features = ['windspeed_log', 'temp_diff', 'temp_humidity']
print("======== Train:")
print(train_df[new_features].describe())
print("======== Test:")
print(test_df[new_features].describe())


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 그래프 스타일 설정
plt.style.use('seaborn')
sns.set_palette("husl")


# 1. 시간대별 자전거 대여량 분석
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='count', data=train_df)
plt.title('Hourly Rental Distribution')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='dayofweek', y='count', data=train_df)
plt.title('Daily Rental Distribution')
plt.show()


plt.figure(figsize=(12, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='RdYlBu_r', center=0)
plt.title('Correlation Matrix')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(x='weather', y='count', data=train_df)
plt.title('Weather Impact on Rentals')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(x='temp', y='count', data=train_df, alpha=0.5)
plt.title('Temperature vs Rental Count')
plt.show()


monthly_rentals = train_df.groupby('month')['count'].mean()
plt.figure(figsize=(12, 6))
monthly_rentals.plot(kind='bar')
plt.title('Average Monthly Rentals')
plt.show()


hourly_rentals = train_df.groupby('hour')['count'].mean()
plt.figure(figsize=(12, 6))
hourly_rentals.plot(kind='line', marker='o')
plt.title('Average Hourly Rentals')
plt.grid(True)
plt.show()


import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# 전처리 완료된 train_df와 test_df가 있다고 가정
# datetime 인덱스 설정 (시간 순서 유지)
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
train_df.set_index('datetime', inplace=True)
train_df.sort_index(inplace=True)

# 특성과 타겟 분리
X = train_df.drop(['casual', 'registered', 'count'], axis=1)
y = train_df['count']

# 시계열 교차 검증 분할 (5개 폴드)
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
    print(f"Fold {fold}:")
    print(f"  Train: {X.iloc[train_index].index.min()} ~ {X.iloc[train_index].index.max()}")
    print(f"  Val  : {X.iloc[val_index].index.min()} ~ {X.iloc[val_index].index.max()}")

# 최종 분할 (마지막 폴드 사용)
train_index, val_index = list(tscv.split(X))[-1]
X_train, X_val = X.iloc[train_index], X.iloc[val_index]
y_train, y_val = y.iloc[train_index], y.iloc[val_index]

print("==========최종 데이터 형태")
print(f"Train: {X_train.shape}, Val: {X_val.shape}")


from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
import matplotlib.pyplot as plt

# 랜덤 포레스트 모델 학습
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = rf_model.predict(X_val)

# RMSLE 계산
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"Random Forest - RMSLE: {rmsle:.4f}, R2: {r2:.4f}")

# 특성 중요도 시각화
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()



# y_true(실제 값) 또는 y_pred(예측 값) 중 하나 이상에 음수 값이 포함 되었는지 확인
print("Minimum y_val:", y_val.min())
print("Minimum y_pred:", y_pred.min())


import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error, r2_score
import matplotlib.pyplot as plt

# XGBoost 모델 초기화
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    random_state=42
)

# 모델 학습
xgb_model.fit(X_train, y_train)

# 검증 세트에 대한 예측
y_pred = xgb_model.predict(X_val)

# 음수 예측값 처리
y_pred = np.maximum(y_pred, 0)

# 평가 지표 계산
rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"XGBoost - RMSLE: {rmsle:.4f}, R2: {r2:.4f}")

# 특성 중요도 시각화
feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.show()


xgb_params = {
    # 트리 관련 파라미터
    'max_depth': [3, 5, 7, 9],        # 트리의 최대 깊이
    'min_child_weight': [1, 3, 5],    # 리프 노드의 최소 가중치 합
    'gamma': [0, 0.1, 0.2],           # 리프 노드를 추가적으로 나눌지 결정하는 최소 손실 감소값
    
    # 부스팅 파라미터
    'learning_rate': [0.01, 0.05, 0.1],  # 학습률
    'n_estimators': [100, 200, 300],     # 트리의 개수
    
    # 샘플링 파라미터
    'subsample': [0.8, 0.9, 1.0],        # 샘플링할 훈련 데이터의 비율
    'colsample_bytree': [0.8, 0.9, 1.0]  # 각 트리마다 샘플링할 피처의 비율
}


from sklearn.model_selection import RandomizedSearchCV

# RandomizedSearchCV 정의
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_params,
    n_iter=100,
    scoring='neg_mean_squared_log_error',  # 수정된 부분
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# 랜덤 서치 수행
random_search.fit(X_train, y_train)

# 최적 파라미터 및 성능 출력
print("Best parameters:", random_search.best_params_)
print("Best RMSLE:", np.sqrt(-random_search.best_score_))


xgb_params = {
    'max_depth': [5, 6, 7, 8], # 트리의 최대 깊이
    'min_child_weight': [1, 3], # 리프 노드의 최소 가중치 합
    'learning_rate': [0.05, 0.1], # 학습률
    'n_estimators': [200, 300], # 트리의 개수
    'subsample': [0.8, 0.9],  # 샘플링할 훈련 데이터의 비율
    'colsample_bytree': [0.8, 0.9] # 각 트리마다 샘플링할 피처의 비율
}



from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=xgb_params,
    n_iter=50,  # 반복 횟수 줄임
    scoring='neg_mean_squared_log_error',
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# 학습 전 데이터 확인
print("Data shape:", X_train.shape)
print("Any NaN in X:", np.isnan(X_train).any())
print("Any NaN in y:", np.isnan(y_train).any())

# 모델 학습
random_search.fit(X_train, y_train)

# 결과 출력
print("\nBest parameters:", random_search.best_params_)
print("Best RMSLE:", np.sqrt(-random_search.best_score_))



xgb_params = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [1, 2, 3],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0, 0.1]
}



from sklearn.metrics import make_scorer
import numpy as np

# 사용자 정의 RMSLE 함수
def custom_rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)  # 음수 예측값을 0으로 변환
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# 사용자 정의 scorer
rmsle_scorer = make_scorer(custom_rmsle, greater_is_better=False)

# RandomizedSearchCV 수정
random_search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=xgb_params,
    n_iter=50,
    scoring=rmsle_scorer,  # 수정된 scorer 사용
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# 모델 학습
random_search.fit(X_train, y_train)

# 결과 출력
print("\nBest parameters:", random_search.best_params_)
print("Best RMSLE:", np.sqrt(-random_search.best_score_))




