import pandas as pd


train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")


# 데이터 크기
train.shape, test.shape


# 데이터 샘플
train.head(2)


# 데이터 샘플
test.head(2)


# 결측값 확인
train.isnull().sum()


# 결측값 확인
test.isnull().sum()


train['count'].hist()


# datetime

train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day

train = train.drop('datetime', axis=1)
test = test.drop('datetime', axis=1)


# test에는 없는 컬럼 삭제
train = train.drop(['casual', 'registered'], axis=1)
train.head(1)


# target 별도 저장
target = train.pop('count')
target


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=2023)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# 평가
from sklearn.metrics import r2_score


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_val)
r2_score(y_val, pred)


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred = rf.predict(X_val)
r2_score(y_val, pred)


from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
pred = xgb.predict(X_val)
r2_score(y_val, pred)


# 하이퍼파라미터 
from xgboost import XGBRegressor
xgb = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=9)
xgb.fit(X_train, y_train)
pred = xgb.predict(X_val)
r2_score(y_val, pred)


# 예측
pred = xgb.predict(test)
pred


# csv 파일 생성
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

submit = pd.DataFrame({
    'datetime': test['datetime'],
    'count': pred
})
submit.to_csv("submission.csv", index=False)


# 점검
print(pd.read_csv("submission.csv"))
print(submit.shape, test.shape)

