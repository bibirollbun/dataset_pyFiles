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


train= pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
data= pd.read_csv('/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv')


train.info()


data.info()


data['Number_of_Ads']= data['Number_of_Ads'].astype('float64')


data.info()


train= train.drop(columns=['id'])


train.columns.to_list()


train= pd.concat([train,data], ignore_index= True)
train= train.drop_duplicates()


train.info()


x_train= train.drop('Listening_Time_minutes', axis= 1)
y_train= train['Listening_Time_minutes']


print(x_train.columns.to_list())


print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")


x_train.isnull().any()


test.columns.to_list()


test= test.drop('id', axis=1)


test['Number_of_Ads']= test['Number_of_Ads'].astype('float64')


num_var= x_train.select_dtypes(include= [np.number]).columns.to_list()
cat_var= x_train.select_dtypes(include= 'object').columns.to_list()



print(num_var)


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

num_trans= Pipeline([
    ('impute', SimpleImputer(strategy= 'median')),
    ('standardscaler', StandardScaler())
])

cat_trans= Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor= ColumnTransformer(transformers=[
    ('num', num_trans, num_var),
    ('cat', cat_trans, cat_var)
], remainder= 'passthrough')


pipeline= Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(n_estimators=10, max_depth=100, random_state=42, verbose=1))
])


imputer = SimpleImputer(strategy='median')
y_train= imputer.fit_transform(y_train.values.reshape(-1, 1)).ravel()


pipeline.fit(x_train, y_train)




