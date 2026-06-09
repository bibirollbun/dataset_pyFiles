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


# ---------------------------------------
# 1. 라이브러리 임포트
# ---------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------
#  데이터 불러오기
# ---------------------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')

# ---------------------------------------
#  데이터 기본 구조 확인
# ---------------------------------------
print("=== Train Data ===")
print(train.shape)
print(train.info())
display(train.head())

print("\n=== Test Data ===")
print(test.shape)
print(test.info())
display(test.head())


# ---------------------------------------
# 기초 통계량 확인 (train)
# ---------------------------------------
print("\n=== train.describe() ===")
display(train.describe(include='all'))

# Price 칼럼을 따로 살펴보기
print("\n=== Price 기초 통계량 ===")
display(train['Price'].describe())


# ---------------------------------------
# 결측치(NaN) 확인
# ---------------------------------------
print("\n=== 결측치 확인 ===")
print(train.isnull().sum())


# ---------------------------------------
# 범주형 변수 분포 확인
# ---------------------------------------

categorical_cols = ['Brand', 'Material','Size', 'Color', 'Style'] 

for col in categorical_cols:
    if col in train.columns:
        print(f"\n=== {col} 상위 10개 값 분포 ===")
        print(train[col].value_counts().head(10))


# ---------------------------------------
#  숫자형 변수 분포 시각화
# ---------------------------------------

numeric_cols = ['Compartments', 'Weight Capacity (kg)'] 

for col in numeric_cols:
    if col in train.columns:
        plt.figure(figsize=(7, 5))
        sns.histplot(train[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()


# ---------------------------------------
#  결측치 처리
# ---------------------------------------

for col in ['Brand', 'Material', 'Size', 'Laptop Compartment',
            'Waterproof', 'Style', 'Color']:
    train[col] = train[col].fillna('Unknown')


median_wcap = train['Weight Capacity (kg)'].median()
train['Weight Capacity (kg)'] = train['Weight Capacity (kg)'].fillna(median_wcap)
















