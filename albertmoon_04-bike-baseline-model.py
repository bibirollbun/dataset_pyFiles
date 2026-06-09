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


import pandas as pd

data_path = '/kaggle/input/bike-sharing-demand/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
submission = pd.read_csv(data_path + 'sampleSubmission.csv')


train.shape, test.shape, submission.shape


# Heavy Rain, Snow, ... 제거
train = train[train['weather'] != 4]


all_data_temp = pd.concat([train, test])
all_data_temp


# train인덱스가 나열되고, 다시 test인덱스가 나열되었으므로
# 실제는 17,378행인데, 마지막 인덱스가 6,492로 보인다.
# 수정해야 한다.
all_data = pd.concat([train, test], ignore_index=True)
all_data


from datetime import datetime


all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])

# datetime 열을 datetime 형식으로 변환 (필요한 경우)
all_data['datetime'] = pd.to_datetime(all_data['datetime'])

# 각각의 구성 요소를 새로운 열로 추가
all_data['year'] = all_data['datetime'].dt.year
all_data['month'] = all_data['datetime'].dt.month
all_data['hour'] = all_data['datetime'].dt.hour

all_data['weekday'] = all_data['date'].apply(lambda x: 
                            datetime.strptime(x, "%Y-%m-%d").weekday())


all_data.head()


drop_features = ['casual', 'registered', 'datetime', 'date', 'month', 'windspeed']

all_data = all_data.drop(drop_features, axis=1)

all_data.head()


X_train = all_data[~pd.isnull(all_data['count'])]
X_test = all_data[pd.isnull(all_data['count'])]

X_train = X_train.drop(['count'], axis=1)
X_test = X_test.drop(['count'], axis=1)

y = train['count']   # 타깃값 


X_train.head()


import numpy as np

def rmsle(y_true, y_pred, convertExp=True):
    # 치우친 데이터를 로그적용으로 정규분포로 변환했으므로
    # 지수변환을 통해 복원해야 한다.
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))

    output = np.sqrt(np.mean((log_true - log_pred)**2))

    return output


from sklearn.linear_model import LinearRegression

linear_reg_model = LinearRegression()


log_y = np.log(y)   # 타깃값 로그 변환
linear_reg_model.fit(X_train, log_y)   # 모델 학습


preds = linear_reg_model.predict(X_train)


print(f'선형 회귀의 RMSLE의 값: {rmsle(log_y, preds, True):.4f}')


linearreg_preds = linear_reg_model.predict(X_test)

submission['count'] = np.exp(linearreg_preds)    # 지수변환
submission.to_csv('submission.csv', index=False)


submission.head()




