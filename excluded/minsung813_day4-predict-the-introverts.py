# 필요 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 불러오기
df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
raw_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# 숫자형 컬럼과 문자형 컬럼 나누기
num_cols = df.select_dtypes(include=['float64', 'int64']).columns #숫자형          
cat_cols = df.select_dtypes(include=['object']).columns.drop('Personality') #문자형

# 숫자형 결측값 평균으로 채우기
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())
    test[col] = test[col].fillna(test[col].mean())

# 문자형 결측값 최빈값으로 채우기
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0]).infer_objects(copy=False)
    test[col] = test[col].fillna(test[col].mode()[0]).infer_objects(copy=False)


# 라벨인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    # 학습 데이터와 테스트 데이터 합쳐서 한 번에 학습
    total = pd.concat([df[col], test[col]], axis=0)
    # 라벨 인코더 학습
    le.fit(total)
    # 학습된 인코더로 변환
    df[col] = le.transform(df[col])
    test[col] = le.transform(test[col])

# Personality도 인코딩하기
df['Personality'] =le.fit_transform(df['Personality'])


# 데이터 학습/검증용 나누기
from sklearn.model_selection import train_test_split
X = df.drop('Personality', axis=1) #입력데이터
y = df['Personality'] #정답

# 학습용 80%, 검증용 20%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 1. 로지스틱 회귀
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
pred1 = model_lr.predict(X_val)
print("LogisticRegression 정확도:", accuracy_score(y_val, pred1))



# 2. XGBoost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_xgb.fit(X_train, y_train)
pred2 = model_xgb.predict(X_val)
print("XGBoost 정확도:", accuracy_score(y_val, pred2))



# 3. 나이브 베이즈
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
pred3 = model_nb.predict(X_val)
print("Naive Bayes 정확도:", accuracy_score(y_val, pred3))



# 테스트 데이터에 예측하기
test_pred = model_lr.predict(test) # model.predict(test) 바꿔주기

# 다시 되돌리기
test_pred_label = le.inverse_transform(test_pred)

# 예측 결과를 DataFrame으로 만들기
submission = pd.DataFrame({
    "id": raw_test['id'],
    "Personality": test_pred_label    
})

# 제출 파일로 저장 (캐글 제출용)
submission.to_csv("submission.csv", index=False)

