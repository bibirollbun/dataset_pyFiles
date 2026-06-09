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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')

# sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


X_train = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
X_test = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')


display(X_train.head(3))
display(X_test.head(3))


# train 데이터 정보
X_train.info()


# test 데이터 정보
X_test.info()


# 수치형 변수 시각화
num_cols = X_train.select_dtypes(include='number').columns

for col in num_cols:
    sns.histplot(X_train[col])
    plt.show()


# 수치형 변수 통계값
X_train.describe(include='number') 

# 평균과 중간값이 비슷 -> 그렇기 때문에 왜도 및 이상치 처리가 필요없다.


X_train.isnull().sum()


X_test.isnull().sum()


corr = X_train.corr(numeric_only='True')
corr


le = LabelEncoder()

X_train['NObeyesdad'] = le.fit_transform(X_train['NObeyesdad']) 



target_corr = X_train.corr(numeric_only='True')['NObeyesdad'].sort_values(ascending=False)
target_corr


X_train['NObeyesdad']


import seaborn
help(seaborn.boxplot)


# 이상치 개수 체크
num_cols = X_train.select_dtypes(include='number').columns

for col in num_cols:
    print(f"{col}")
    sns.boxplot(data=X_train[col])
    q1 = X_train[col].quantile(.25)
    q3 = X_train[col].quantile(.75)
    IQR = q3 - q1
    lower = q1 - 1.5 * IQR
    upper = q3 + 1.5 * IQR
    outliers = X_train[(X_train[col] < lower) | (X_train[col] > upper)]
    print(f"{col}: {len(outliers)}")
    plt.show()

    


# Age, NCP
X_train['Age'].describe()


X_train['NCP'].describe()


cat_cols = X_train.select_dtypes(include='object').columns
cat_cols


# 범주형 변수에 따른 target값 분포 비교
target_cols = 'NObeyesdad'

for col in cat_cols:
    sns.countplot(data=X_train, x=col, hue=target_cols)
    plt.show()


X_train.info()


# 범주형 변수, Label Encoder은 1개의 컬럼씩 가능
cat_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
       'SCC', 'CALC', 'MTRANS']

for col in cat_cols:
    le = LabelEncoder()

    le.fit(pd.concat([X_train[col], X_test[col]]))
    
    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])


X_train


X_test


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


X_train


y_train = X_train['NObeyesdad']
X_train = X_train.drop(columns=['NObeyesdad', 'id'])


y_train


rf_model = RandomForestClassifier(random_state=42)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

grid_rf = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=3, n_jobs=-1, scoring='accuracy')
grid_rf.fit(X_train, y_train)

# 결과 확인
print("RandomForest Best Params:", grid_rf.best_params_)
print("RandomForest CV Score:", grid_rf.best_score_)


X_test = X_test.drop(columns='id')


X_test


best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)


type(y_pred_rf[:10])


a = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')

a[10:20]


a = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')
a = a['id']


y_train.unique()


label_map = {0:'Insufficient_Weight', 1: 'Normal_Weight', 2: 'Obesity_Type_I', 3: 'Obesity_Type_II', 4:'Obesity_Type_III', 5: 'Overweight_Level_I', 6: 'Overweight_Level_II'}


type(y_pred_rf)


pred = y_pred_rf.tolist()


y_pred_rf = []
for p in pred:
    # print(p)
    y_pred_rf.append(label_map[p])


y_pred_rf[:10]


print(a.shape)


submit = pd.DataFrame(
    {
        'id': a,
        'NObeyesdad': y_pred_rf
    }
)


submit.to_csv('./submission.csv', index=False)




