import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


train.describe(include='all')


train.isnull().sum()


print(train.shape)
print(train.head(3))
print("\n",train.info())
print("\n",train.columns)


# target 변수 분리
target = train.pop('Personality')


# id값은 필요없음 : 고유값이라 예측 시 의미 없음
train = train.drop(columns='id') # 학습에 필요없는 train id 는 삭제
test_id = test.pop('id') # 결과 csv에 필요한 test id는 살려준다. 

# 결측치가 전부 10퍼센트 미만이기 때문에 수치형:평균, 범주형:최빈 으로 채워준다.
object_cols = train.select_dtypes(include='object').columns 
numeric_cols = train.select_dtypes(exclude='object').columns
# 범주형 최빈값 대체
for col in object_cols: 
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])
# 수치형 평균 대체
for col in numeric_cols:
    train[col] = train[col].fillna(train[col].mean())
    test[col] = test[col].fillna(test[col].mean())
    
# 범주형 변수, 수치형 변수 시각화하여 분포 살피기 : 스케일링 여부 판단
import matplotlib.pyplot as plt

for col in object_cols:
    test[col].value_counts().plot(kind="bar")
    plt.title(f"Value counts of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

for col in numeric_cols:
    test[col].plot(kind="hist", bins=20)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# 스케일링은 필요없어보인다. 바~로 인코딩
# pd.get_dummies() 괄호 속 데이터프레임을 넣으면 알아서 범주형 변수들을 one-hot encoding하여 반환합니다.
# 가장 쉬운 인코딩 방법
train = pd.get_dummies(train)
test = pd.get_dummies(test)


# 바로 모델을 불러와 예측해도 되지만 지난번에 배웠던 train_test_split 을 통한 validation 과정을 넣어봅니다. 

# 1: validation모듈, 분류모델, 평가지표를 각각 import
from sklearn.model_selection import train_test_split # validation
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트 분류
from sklearn.linear_model import LogisticRegression # 로지스틱회귀 분류
from sklearn.metrics import accuracy_score, f1_score # 분류 평가지표

# 2: train 데이터를 잘라서 validation 데이터로 만들기
X_tr, X_val, y_tr, y_val = train_test_split(train, target, test_size=0.2, random_state=2025) # train데이터 중 20%를 validation 데이터로 잘라냅니다.

# 3: validation
rf = RandomForestClassifier()
lr = LogisticRegression()

rf.fit(X_tr, y_tr)
lr.fit(X_tr, y_tr)

rf_pred = rf.predict(X_val)
lr_pred = lr.predict(X_val)

# 4: validation evaluation : 예측 결과들(예측값)과 y_val(실제값)을 통해 모델(rf, lr)의 성능을 비교합니다. 
print(f"랜덤포레스트 accuracy score: {accuracy_score(y_val, rf_pred)}")
print(f"랜덤포레스트 f1 score: {f1_score(y_val, rf_pred, average='macro')}") # 정석대로면 target변수도 수치형(1/0)으로 바꿔줘햐 하지만
print(f"로지스틱회귀 accuracy score: {accuracy_score(y_val, lr_pred)}")
print(f"로지스틱회귀 f1 score: {f1_score(y_val, lr_pred, average='macro')}") # macro 값을 명시해주면 범수형(etrovert, introvert)상태로 평가 가능

# 5: model select : 성능 좋은 모델을 선택한다 : 로지스틱회귀가 약간 더 좋은 성능! 
# 랜덤포레스트 accuracy score: 0.9643724696356275
# 랜덤포레스트 f1 score: 0.9531554236873825
# 로지스틱회귀 accuracy score: 0.9657219973009447
# 로지스틱회귀 f1 score: 0.954851327027574
lr_pred_test = lr.predict(test) # 최적 모델로 predict

# 6: csv로 만들어주기
result = pd.DataFrame({'id':test_id,
                      'Personality':lr_pred_test})

result.to_csv('submission.csv', index=False)


pd.read_csv('submission.csv')

