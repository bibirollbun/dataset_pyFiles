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


import os

# 실제 파일 경로 출력해보기
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pandas as pd

# 올바른 경로로 수정된 코드
train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')

# 데이터 미리 보기
train.head()


print(train.columns)


from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 입력값(X)과 정답(y) 나누기
X = train.drop(['id', 'rainfall'], axis=1)
y = train['rainfall']

# 결측값을 평균값으로 채우기 (이게 핵심!)
X = X.fillna(X.mean())

# 모델 만들고 학습하기
model = RandomForestClassifier()
model.fit(X, y)

# 테스트 데이터도 같은 방식으로 처리
X_test = test.drop(['id'], axis=1)
X_test = X_test.fillna(X_test.mean())  # 결측값 채우기

# 예측하기
predictions = model.predict(X_test)

# 제출 파일 만들기
sample_submission['expected'] = predictions
sample_submission.to_csv('submission.csv', index=False)


print("train 데이터 크기:", train.shape)
print("test 데이터 크기:", test.shape)


# 중복된 행 수 확인
duplicate_count = train.duplicated().sum()
print("중복된 행 수:", duplicate_count)


# 숫자형 데이터에 대한 통계 요약
train.describe()


train.mean(numeric_only=True)


# 각 칼럼의 데이터 타입 확인
print(train.dtypes)


train.select_dtypes(include=['number']).columns


train.select_dtypes(include=['object']).columns


categorical_cols = train.select_dtypes(include=['object']).columns
print("범주형 컬럼들:", list(categorical_cols))


# 각 범주형 컬럼의 고유값 보기
for col in categorical_cols:
    print(f"\n[{col}] 고유값들:")
    print(train[col].unique())


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(train.isnull(), cbar=False, cmap="YlGnBu")
plt.title("Missing Values in Train Dataset")  # 영어로 바꿔서 경고 제거
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(train.isnull(), cbar=False, cmap="YlGnBu")
plt.title("Train 데이터의 결측값 시각화")
plt.show()


numeric_cols = train.select_dtypes(include=['number']).columns.tolist()

train[numeric_cols].hist(figsize=(15, 10), bins=30)
plt.suptitle("수치형 컬럼들의 분포 (히스토그램)", fontsize=16)
plt.show()


plt.figure(figsize=(15, 6))
sns.boxplot(data=train[numeric_cols], orient="h")
plt.title("수치형 변수들의 이상치 분포 (Boxplot)")
plt.show()

