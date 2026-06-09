# 필요 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 불러오기
df = pd.read_csv('/kaggle/input/playground-series-s4e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')
raw_test = pd.read_csv('/kaggle/input/playground-series-s4e4/test.csv')


# 숫자형 컬럼과 문자형 컬럼 나누기
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop('Rings') #숫자형          
cat_cols = df.select_dtypes(include=['object']).columns #문자형

# 이상치 처리 -> # Height == 0 값을 평균으로 대체 (test만)
mean_height = df[df["Height"] != 0]["Height"].mean()
test.loc[test["Height"] == 0, "Height"] = mean_height


# 라벨 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
test['Sex'] = le.transform(test['Sex'])


# 데이터 학습/검증용 나누기
from sklearn.model_selection import train_test_split
X = df.drop('Rings', axis=1) #입력데이터
y = df['Rings'] #정답

# 학습용 80%, 검증용 20%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. 선형 회귀
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred1 = model_lr.predict(X_val)

print("LinearRegression MSE:", mean_squared_error(y_val, pred1))


# 2. 랜덤 포레스트 회귀
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
pred2 = model_rf.predict(X_val)

print("RandomForestRegressor MSE:", mean_squared_error(y_val, pred2))


# 3. XGBoost 회귀
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)
pred3 = model_xgb.predict(X_val)

print("XGBRegressor MSE:", mean_squared_error(y_val, pred3))


# 테스트 데이터 예측
test_pred = model_xgb.predict(test)  # 사용 중인 모델로 예측

# 제출 파일 생성
submission = pd.DataFrame({
    "id": raw_test["id"],
    "Rings": test_pred  # 예측 결과 그대로 사용
})

# CSV 파일로 저장
submission.to_csv("submission.csv", index=False)

