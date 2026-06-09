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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor


train_data = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/train_data.csv")
test_data = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/test_data.csv")
sample_solution = pd.read_csv("/kaggle/input/aviachipta-narxini-bashorat-qilish/sample_solution.csv")


train_data.head()


train_data.info()


train_data.describe()


train_x=train_data.drop('price',axis=1)
train_y=train_data['price'].copy()


train_set,test_set=train_test_split(train_data,test_size=0.2,random_state=42)


num = ['id', 'duration', 'days_left']
cat = ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']


num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
])


train_prepared = full_pipeline.fit_transform(train_x)


RF_model = RandomForestRegressor()
RF_model.fit(train_prepared, train_y)


test_x=test_set.drop('price',axis=1)
test_y=test_set['price']


test_prepared = full_pipeline.transform(test_x)


final_result=RF_model.predict(test_prepared)


score_data = mean_absolute_error(final_result, test_y) 
print(score_data)


prepared_test_data=full_pipeline.transform(test_data)


predicted_test_data=RF_model.predict(prepared_test_data)
print(predicted_test_data)


solution_file=pd.DataFrame({'id':test_data['id'],'price':predicted_test_data})
solution_file.head()


solution_file.to_csv('my_submission',index=False)




