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
from sklearn.ensemble import RandomForestClassifier


train_data_path = '../input/mlolympiadbd2024/train.csv'
train_data = pd.read_csv(train_data_path)


train_data.head()


y = train_data.RainToday
rain_features = [
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm"
]
X = train_data[rain_features]

# handling strings
X = pd.get_dummies(X, columns=['WindDir9am', 'WindDir3pm', 'WindGustDir'])



print(X.isnull().sum())
print('--')
print((X.dtypes == 'object').sum())
print('--')
print(X.select_dtypes(include='object').apply(lambda col: col.notnull().sum()))


rain_model = RandomForestClassifier(random_state = 1)

rain_model.fit(X, y)


test_data = pd.read_csv('../input/mlolympiadbd2024/test.csv')


test_X = test_data[rain_features]
# handling strings
test_X = pd.get_dummies(test_X, columns=['WindDir9am', 'WindDir3pm', 'WindGustDir'])


rain_predicts = rain_model.predict(test_X)


output = pd.DataFrame({
    "ID":test_data.ID,
    "prediction":rain_predicts
})
output.to_csv('submission.csv', index=False)

