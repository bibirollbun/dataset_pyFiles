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


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')



print(train.shape) ## Train 데이터 크기
print(train.head()) ## 데이터의 기본 구조


duplicate_count = train.duplicated().sum()
print(f" 중복된 행 개수: {duplicate_count}")

if ( duplicate_count > 0 ) :
    train = train.drop_duplicates()
    print(f" 중복된 행 개수: {duplicate_count}")


# 수치형 변수만 추출
num_cols = train.select_dtypes(include='number').columns

# 이상치 개수 저장용
outlier_count = {}

for col in num_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # 이상치 개수 계산
    outliers = ((train[col] < lower) | (train[col] > upper)).sum()
    outlier_count[col] = outliers

# 결과를 DataFrame으로 정리하고 정렬
outlier_df = pd.DataFrame(list(outlier_count.items()), columns=['column', 'outlier_count'])
outlier_df = outlier_df.sort_values(by='outlier_count', ascending=False)

# 가장 이상치가 많은 컬럼
outlier_column = outlier_df.iloc[0]['column']
print('가장 이상치가 많은 컬럼 : ' + outlier_column)

# 가장 이상치가 적은 컬럼
min_outlier_column = outlier_df.iloc[-1]['column']
print('가장 이상치가 적은 컬럼 : ' + outlier_column)



import matplotlib.pyplot as plt

plt.boxplot(train[outlier_column])
plt.title('Column with the most outliers : ' + outlier_column)
plt.xlabel(outlier_column)
plt.ylabel('value')
plt.show()

plt.boxplot(train[min_outlier_column])
plt.title('Column with the fewest outliers :' + min_outlier_column)
plt.xlabel(min_outlier_column)
plt.ylabel('value')
plt.show()


missing_values = train.isnull().sum()
print(missing_values) 
train.fillna(train.mean(), inplace=True)


train.describe()


numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns
print("수치형 컬럼 목록:")
print(numeric_cols.tolist())

categorical_cols = train.select_dtypes(include=['object', 'category']).columns
print("범주형 컬럼 목록:")
print(categorical_cols.tolist())


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

