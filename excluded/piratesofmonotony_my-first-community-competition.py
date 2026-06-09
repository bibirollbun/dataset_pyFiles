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


sample=pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/sample_submission.csv",sep=",")
sample


train=pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/train.csv",sep=",")


test=pd.read_csv("/kaggle/input/russian-car-plates-prices-prediction/test.csv",sep=",")


train.shape,test.shape


train.head(5)


test.head(5)


y=train['price']


train.drop(columns=['id','price'],axis=1,inplace=True)


test.drop(columns=['id','price'],axis=1,inplace=True)


train



train['Letters'] = train['plate'].str.findall(r'[A-Za-z]').str.join('').str.lower()

print(train)



test['Letters'] = test['plate'].str.findall(r'[A-Za-z]').str.join('').str.lower()

print(test)


test.isna().sum()


train.isna().sum()



df=pd.DataFrame()
# Convert column to datetime format
df['datetime'] = pd.to_datetime(train['date'])

# Extract components
train['date'] = df['datetime'].dt.day
train['day_number'] = df['datetime'].dt.dayofweek + 1  # Monday = 1, Sunday = 7
train['hour'] = df['datetime'].dt.hour
train['minute'] = df['datetime'].dt.minute
train['second'] = df['datetime'].dt.second
train['week_number'] = df['datetime'].dt.isocalendar().week
train['month'] = df['datetime'].dt.month
train['year'] = df['datetime'].dt.year

train



tf=pd.DataFrame()
# Convert column to datetime format
tf['datetime'] = pd.to_datetime(test['date'])

# Extract components
test['date'] = tf['datetime'].dt.day
test['day_number'] = tf['datetime'].dt.dayofweek + 1  # Monday = 1, Sunday = 7
test['hour'] = tf['datetime'].dt.hour
test['minute'] = tf['datetime'].dt.minute
test['second'] = tf['datetime'].dt.second
test['week_number'] = tf['datetime'].dt.isocalendar().week
test['month'] = tf['datetime'].dt.month
test['year'] = tf['datetime'].dt.year

test


from sklearn.preprocessing import OrdinalEncoder
en=OrdinalEncoder()
train['Letters']=en.fit_transform(train[['Letters']])
test['Letters']=en.transform(test[['Letters']])
train=pd.DataFrame(train)
test=pd.DataFrame(test)


train


import re
def extract_info(plate):
    match = re.match(r'([ABEKMHOPCTYX]{1}[0-9]{3}[ABEKMHOPCTYX]{2})(\d{2,3}[A-Z]?)$', plate)
    if match:
        return match.group(1)[1:4], match.group(2)  # Extracting 3-digit number & region
    return None, None  # Return None if no match

# Apply extraction function
train[['number', 'region']] = train['plate'].apply(lambda x: pd.Series(extract_info(x)))

train=pd.DataFrame(train)
train.drop(columns=['plate'],axis=1,inplace=True)





# Extraction function
def extract_info(plate):
    match = re.match(r'([ABEKMHOPCTYX]{1}[0-9]{3}[ABEKMHOPCTYX]{2})(\d{2,3}[A-Z]?)$', plate)
    if match:
        return match.group(1)[1:4], match.group(2)  # Extracting 3-digit number & region
    return None, None  # Return None if no match

# Apply extraction function
test[['number', 'region']] = test['plate'].apply(lambda x: pd.Series(extract_info(x)))

# Drop 'plate' column
test.drop(columns=['plate'], axis=1, inplace=True)
test=pd.DataFrame(test)

print(test)



train


from sklearn.preprocessing import RobustScaler
import pandas as pd
scaler = RobustScaler()
scaled_data = scaler.fit_transform(train)
tf=scaler.transform(test)
df= pd.DataFrame(scaled_data, columns=train.columns)
tf=pd.DataFrame(tf,columns=test.columns)
tf




from xgboost import XGBRegressor
# model=XGBRegressor()
from sklearn.linear_model import LinearRegression
# model=LinearRegression()


from catboost import CatBoostRegressor
model=CatBoostRegressor()


model.fit(df,y)


model.score(df,y)


p=model.predict(df)


prediction=model.predict(tf)


import numpy as np

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_value = np.mean( np.abs(y_pred - y_true) / denominator) * 100
    return smape_value


smape_score = smape(y,p)
print("SMAPE Score:", smape_score)




test['id']=[51636+x for x in range(len(test))]
test['price']=prediction
test[['id', 'price']].to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully with two columns!")



# Read and print the submission file
submission = pd.read_csv('submission.csv')
print(submission)





